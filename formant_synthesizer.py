# %%
import numpy as np
import scipy.signal as sig
import soundfile as sf
import matplotlib.pyplot as plt

# %%
def generate_sawtooth(F0, sample_rate, duration):
    # Create a time vector with the proper number of samples.
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Generate the sawtooth waveform. The function returns values in the range [-1, 1].
    signal = sig.sawtooth(2 * np.pi * F0 * t)
    return signal

# %%
# --- Formant Resonator ---
# y[n] = A*x[n-1] + B*y[n-1] + C*y[n-2]
# with A = 1 – B – C, B = 2 exp(–π F_B T_s) cos(2π F T_s), C = -exp(–2π F_B T_s)
def formant_resonator(signal, formant_frequency, sample_rate, bandwidth):
    T_s = 1 / sample_rate
    B_coef = 2 * np.exp(-np.pi * bandwidth * T_s) * np.cos(2 * np.pi * formant_frequency * T_s)
    C_coef = -1*np.exp(-2 * np.pi * bandwidth * T_s)
    A_coef = 1 - B_coef - C_coef
    y = np.zeros_like(signal)
    # For n = 0, no previous input; set y[0] = 0.
    y[0] = 0
    # For n = 1, use x[0]
    if len(signal) > 1:
        y[1] = A_coef * signal[0] + B_coef * y[0]
    # For n >= 2, follow the difference equation.
    for n in range(2, len(signal)):
        y[n] = A_coef * signal[n-1] + B_coef * y[n-1] + C_coef * y[n-2]
    # Return both the time response and H (for TF plotting)
    f = np.linspace(0, sample_rate/2, 5000)
    theta = 2 * np.pi * f * T_s  # note: T_s cancels, but we keep for clarity
    # For the formant resonator, the transfer function is:
    # H(f) = (A * exp(-j*theta)) / (1 - B exp(-j*theta) - C exp(-j*2*theta))
    H = (A_coef * np.exp(-1j * theta)) / (1 - B_coef * np.exp(-1j * theta) - C_coef * np.exp(-1j * 2 * theta))

    return y, np.array(H)

# %%
# Plot from H(z)
def plot_response(H, sample_rate, label):
    f = np.linspace(0, sample_rate/2, len(H))
    amplitude = 20 * np.log10(np.abs(H))
    phase = np.unwrap(np.angle(H))

    plt.figure(figsize=(8,6))
    plt.subplot(2,1,1)
    plt.plot(f, amplitude, 'b')
    plt.title(f"{label} - Amplitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(f, phase, 'r')
    plt.title(f"{label} - Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
def plot_response_multiple(H_list, sample_rate, label_list=None):
    # Use the length of the first H to build a frequency vector.
    N = len(H_list[0])
    f = np.linspace(0, sample_rate / 2, N)
    
    plt.figure(figsize=(10, 8))
    
    # Plot amplitude responses.
    plt.subplot(2, 1, 1)
    for i, H in enumerate(H_list):
        eps = 1e-12
        amplitude = 20 * np.log10(np.abs(H) + eps)
        label = label_list[i] if label_list is not None and i < len(label_list) else f"Filter {i+1}"
        plt.plot(f, amplitude, label=label)
    plt.title("Amplitude Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True)
    plt.legend()
    
    # Plot phase responses.
    plt.subplot(2, 1, 2)
    for i, H in enumerate(H_list):
        phase = np.unwrap(np.angle(H))
        label = label_list[i] if label_list is not None and i < len(label_list) else f"Filter {i+1}"
        plt.plot(f, phase, label=label)
    plt.title("Phase Response")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# %%
fs = 10000 # sampling rate
length_seconds = 2 # generate 2s signal
length_samples = fs * length_seconds # how many samples there are in the 2s audio
f0 = 100

# %%
# Generate a voice source
signal = generate_sawtooth(f0, fs, length_seconds)

# %%
vowel = 'i'
match vowel:
    case 'i':
        center_frequency = [310, 2020, 2960, 3800, 4800] # Formants of the i vowel
        BW = [45, 200, 400, 100, 150] # bandwidths of the formants
    case 'a':
        center_frequency = [660, 1200, 2550, 3500, 4500] # Formants of the a vowel
        BW = [100, 70, 200, 100, 150] # bandwidths of the formants
    case 'u':
        center_frequency = [350, 1250, 2200, 3500, 4500] # Formants of the u vowel
        BW = [65, 110, 140, 100, 150] # bandwidths of the formants
    case 'schwa':
        center_frequency = [500, 1500, 2500, 3500, 4500] # Formants of the schwa vowel
        BW = [100, 100, 100, 100, 100] # bandwidths of the formants
    case _:
        center_frequency = [500, 1500, 2500, 3500, 4500] # Formants of the schwa vowel
        BW = [100, 100, 100, 100, 100] # bandwidths of the formants


# %%
# The individual formant responses to be added together in the parallel synthesis process
first_rez, first_H = formant_resonator(signal, center_frequency[0], fs, BW[0])
second_rez, second_H = formant_resonator(signal, center_frequency[1], fs, BW[1])
third_rez, third_H = formant_resonator(signal, center_frequency[2], fs, BW[2])
fourth_rez, fourth_H = formant_resonator(signal, center_frequency[3], fs, BW[3])
fifth_rez, fifth_H = formant_resonator(signal, center_frequency[4], fs, BW[4])

# Cascade synthesis, each previous output is the new input, the first_rez from the parallel is also the first_rez for the cascade
second_cascade_rez, second_cascade_H = formant_resonator(first_rez, center_frequency[1], fs, BW[1])
third_cascade_rez, third_cascade_H = formant_resonator(second_rez, center_frequency[2], fs, BW[2])
fourth_cascade_rez, fourth_cascade_H = formant_resonator(third_rez, center_frequency[3], fs, BW[3])
fifth_cascade_rez, fifth_cascade_H = formant_resonator(fourth_rez, center_frequency[4], fs, BW[4])

# %%
# The amplitude values of the formant peaks for individual formants
# Calculate the amplitude response in linear space and find the amplitude value 
# for the given formant at the center frequency so gain can be calculated later
first_formant_amplitude_linear = np.abs(first_H)
single_rez_first_amplitude_linear = first_formant_amplitude_linear[center_frequency[0]]

second_formant_amplitude_linear = np.abs(second_H)
single_rez_second_amplitude_linear = second_formant_amplitude_linear[center_frequency[1]]

third_formant_amplitude_linear = np.abs(third_H)
single_rez_third_amplitude_linear = third_formant_amplitude_linear[center_frequency[2]]

fourth_formant_amplitude_linear = np.abs(fourth_H)
single_rez_fourth_amplitude_linear = fourth_formant_amplitude_linear[center_frequency[3]]

fifth_formant_amplitude_linear = np.abs(fifth_H)
single_rez_fifth_amplitude_linear = fifth_formant_amplitude_linear[center_frequency[4]]

# %%
# To connect resonator is cascade in the frequency domain the transfer functions H are multiplied together.
# In the time dimension (to get the output) they are convolved.
# There is no need to do that explicitly considering that passing a signal through a filter is the same as convolving them.
cascade_H = first_H * second_H * third_H * fourth_H * fifth_H
cascade_output = fifth_cascade_rez

# Demonstration of fewer formants in cascade
cascade_H_3 = first_H * second_H * third_H
# Normalize the output values so they fall between 1 and -1
cascade_output = cascade_output / (np.max(np.abs(cascade_output)) + 1e-12)

# Turn the output to audio
sf.write(f'{vowel}_cascade_output.flac', cascade_output, fs)

# %%
cascade_amplitude_linear = np.abs(cascade_H)
# The aplitude values at the center frequency of each resonator
cascade_first_amplitude_linear = cascade_amplitude_linear[center_frequency[0]]
cascade_second_amplitude_linear = cascade_amplitude_linear[center_frequency[1]]
cascade_third_amplitude_linear = cascade_amplitude_linear[center_frequency[2]]
cascade_fourth_amplitude_linear = cascade_amplitude_linear[center_frequency[3]]
cascade_fifth_amplitude_linear = cascade_amplitude_linear[center_frequency[4]]

# Turn it to dB to compare to the ones in parallel in the results section
# Put the file path where you want the .txt saved
cascade_dB_f1 = 20 * np.log10(cascade_first_amplitude_linear)
cascade_dB_f2 = 20 * np.log10(cascade_second_amplitude_linear)
cascade_dB_f3 = 20 * np.log10(cascade_third_amplitude_linear)
cascade_dB_f4 = 20 * np.log10(cascade_fourth_amplitude_linear)
cascade_dB_f5 = 20 * np.log10(cascade_fifth_amplitude_linear)
print(cascade_dB_f1, cascade_dB_f2, cascade_dB_f3, cascade_dB_f4, cascade_dB_f5)
with open(f"{vowel}.txt", "w") as file:
    file.write("Amplitude values for cascade synthesizer:\n")
    file.write(f"F1 = {cascade_dB_f1:.4f}\n")
    file.write(f"F2 = {cascade_dB_f2:.4f}\n")
    file.write(f"F3 = {cascade_dB_f3:.4f}\n")
    file.write(f"F4 = {cascade_dB_f4:.4f}\n")
    file.write(f"F5 = {cascade_dB_f5:.4f}\n")

# %%
# Calculate the gain by dividing the amplitude values from the cascade synthesiser with the individual resonator amplitude values
G1_lin = cascade_first_amplitude_linear / single_rez_first_amplitude_linear
G2_lin = cascade_second_amplitude_linear / single_rez_second_amplitude_linear
G3_lin = cascade_third_amplitude_linear / single_rez_third_amplitude_linear
G4_lin = cascade_fourth_amplitude_linear / single_rez_fourth_amplitude_linear
G5_lin = cascade_fifth_amplitude_linear / single_rez_fifth_amplitude_linear

# %%
# The transfer functions
# Every other gain changes sign to implement alternating polarity
# Three resonators
three_parallel_H = G1_lin * first_H - G2_lin * second_H + G3_lin * third_H
three_parallel_H_no_alt_polarity = G1_lin * first_H + G2_lin * second_H + G3_lin * third_H
# Four resonators
four_parallel_H = G1_lin * first_H - G2_lin * second_H + G3_lin * third_H - G4_lin * fourth_H
# Five resonators
five_parallel_H = G1_lin * first_H - G2_lin * second_H + G3_lin * third_H - G4_lin * fourth_H + G5_lin * fifth_H

# Amplitude values at formant peaks
three_amplitude = []
for i in range(3):
    amplitude = 20 * np.log10(np.abs(three_parallel_H[center_frequency[i]]))
    three_amplitude.append(amplitude)
# Put the file paths where you want the .txt saved
print(three_amplitude)
with open(f"{vowel}.txt", "w") as file:
    file.write("Amplitude values for cascade synthesizer:\n")
    for i in range(len(three_amplitude)):
        file.write(f"F{i} = {three_amplitude[i]:.4f}\n")
    

four_amplitude = []
for i in range(4):
    amplitude = 20 * np.log10(np.abs(four_parallel_H[center_frequency[i]]))
    four_amplitude.append(amplitude)
    
print(four_amplitude)
with open(f"{vowel}.txt", "w") as file:
    file.write("Amplitude values for cascade synthesizer:\n")
    for i in range(len(four_amplitude)):
        file.write(f"F{i} = {four_amplitude[i]:.4f}\n")

five_amplitude = []
for i in range(5):
    amplitude = 20 * np.log10(np.abs(five_parallel_H[center_frequency[i]]))
    five_amplitude.append(amplitude)
    
print(five_amplitude)
with open(f"{vowel}", "w") as file:
    file.write("Amplitude values for cascade synthesizer:\n")
    for i in range(len(five_amplitude)):
        file.write(f"F{i} = {five_amplitude[i]:.4f}\n")


# %%
# The outputs

# The output without spectral shaping
# Three resonators
three_parallel_output_lin = G1_lin * first_rez - G2_lin * second_rez + G3_lin * third_rez
# Four resonators
four_parallel_output_lin = G1_lin * first_rez - G2_lin * second_rez + G3_lin * third_rez - G4_lin * fourth_rez
# Five resonators
five_parallel_output_lin = G1_lin * first_rez - G2_lin * second_rez + G3_lin * third_rez - G4_lin * fourth_rez + G5_lin * fifth_rez

# Normalize the output to be between -1 and 1 (add 1e-12 to avoid dividing by zero)
# Three
three_parallel_output_lin = three_parallel_output_lin / np.max(np.abs(three_parallel_output_lin)) + 1e-12
# Four
four_parallel_output_lin = four_parallel_output_lin / np.max(np.abs(four_parallel_output_lin)) + 1e-12
# Five
five_parallel_output_lin = five_parallel_output_lin / np.max(np.abs(five_parallel_output_lin)) + 1e-12
# Turn the output into a sound file
sf.write(f'{vowel}_3.flac', three_parallel_output_lin, fs)

sf.write(f'{vowel}_4.flac', four_parallel_output_lin, fs)

sf.write(f'{vowel}_5.flac', five_parallel_output_lin, fs)

# %%
# Plot individual formant responses
plot_response(first_H, fs, 'first formant response')
plot_response(second_H, fs, 'second formant response')
plot_response(third_H, fs, 'third formant response')
plot_response(fourth_H, fs, 'fourth formant response')
plot_response(fifth_H, fs, 'fifth formant response')
plot_response_multiple([cascade_H, first_H, second_H, third_H, fourth_H, fifth_H], fs, ['Cascade', 'F1', 'F2', 'F3', 'F4', 'F5'])

# Plot the cascade response
plot_response(cascade_H, fs, 'cascade response')

# Three resonator parallel responses
plot_response(three_parallel_H, fs, 'Three resonator parallel -- no spectral shaping')
plot_response_multiple([cascade_H, three_parallel_H], fs, ['cascade', '3 formant parallel'])
plot_response_multiple([cascade_H, three_parallel_H, three_parallel_H_no_alt_polarity], fs, ['Cascade', 'Parallel -- alternating polarity', 'Parallel -- constant polarity'])

# Four resonator parallel responses
plot_response(four_parallel_H, fs, 'Four resonator parallel -- no spectral shaping')
plot_response_multiple([cascade_H, four_parallel_H], fs, ['cascade', '4 formant parallel'])

# Five resonator parallel responses
plot_response(five_parallel_H, fs, 'Five resonator parallel -- no spectral shaping')
plot_response_multiple([cascade_H, five_parallel_H], fs, ['cascade', '5 formant parallel'])

# Demonstration of fewer formants in cascade
plot_response_multiple([cascade_H, cascade_H_3, three_parallel_H], fs, ['Cascade -- 5 formants', 'Cascade -- 3 formants', 'Parallel -- 3 formants'])
plot_response_multiple([cascade_H, cascade_H_3], fs, ['Cascade -- 5 formants', 'Cascade -- 3 formants'])

# Parallel combined
plot_response_multiple([three_parallel_H, cascade_H], fs, ['Parallel -- 3 formants', 'Cascade'])
plot_response_multiple([three_parallel_H, four_parallel_H, five_parallel_H, cascade_H], fs, ['Parallel -- 3 formants', 'Parallel -- 4 formants', 'Parallel -- 5 formants', 'Cascade'])


