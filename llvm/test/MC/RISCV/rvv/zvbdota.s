# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvqwbdota8i,+experimental-zvqwbdota16i,+experimental-zvfwbdota16bf,+experimental-zvfqwbdota8f,+experimental-zvfbdota32f %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvqwbdota8i,+experimental-zvqwbdota16i,+experimental-zvfwbdota16bf,+experimental-zvfqwbdota8f,+experimental-zvfbdota32f %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvqwbdota8i,+experimental-zvqwbdota16i,+experimental-zvfwbdota16bf,+experimental-zvfqwbdota8f,+experimental-zvfbdota32f --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vsetvli zero, zero, e8alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x70,0x00,0x1c]
vsetvli zero, zero, e8alt, m1, ta, ma

# CHECK-INST: vsetvli zero, zero, e16alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x70,0x80,0x1c]
vsetvli zero, zero, e16alt, m1, ta, ma

# Integer Batched Dot-Product Instructions

# CHECK-INST: vqwbdotau.vv v8, v16, v12, 1
# CHECK-ENCODING: [0x77,0x04,0x16,0xbb]
# CHECK-ERROR: instruction requires the following: 'Zvqwbdota8i' or 'Zvqwbdota16i' (8-bit or 16-bit integer batched dot-product extension){{$}}
vqwbdotau.vv v8, v16, v12, 1

# CHECK-INST: vqwbdotau.vv v8, v16, v12, 2, v0.t
# CHECK-ENCODING: [0x77,0x04,0x26,0xb9]
# CHECK-ERROR: instruction requires the following: 'Zvqwbdota8i' or 'Zvqwbdota16i' (8-bit or 16-bit integer batched dot-product extension){{$}}
vqwbdotau.vv v8, v16, v12, 2, v0.t

# CHECK-INST: vqwbdotas.vv v8, v16, v12, 1
# CHECK-ENCODING: [0x77,0x04,0x16,0xbf]
# CHECK-ERROR: instruction requires the following: 'Zvqwbdota8i' or 'Zvqwbdota16i' (8-bit or 16-bit integer batched dot-product extension){{$}}
vqwbdotas.vv v8, v16, v12, 1

# CHECK-INST: vqwbdotas.vv v8, v16, v12, 2, v0.t
# CHECK-ENCODING: [0x77,0x04,0x26,0xbd]
# CHECK-ERROR: instruction requires the following: 'Zvqwbdota8i' or 'Zvqwbdota16i' (8-bit or 16-bit integer batched dot-product extension){{$}}
vqwbdotas.vv v8, v16, v12, 2, v0.t

# BF16 Batched Dot-Product Instructions

# CHECK-INST: vfwbdota.vv v8, v16, v12, 1
# CHECK-ENCODING: [0x77,0x14,0x16,0xb3]
# CHECK-ERROR: instruction requires the following: 'Zvfwbdota16bf' (BF16 batched dot-product extension){{$}}
vfwbdota.vv v8, v16, v12, 1

# CHECK-INST: vfwbdota.vv v8, v16, v12, 2, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xb1]
# CHECK-ERROR: instruction requires the following: 'Zvfwbdota16bf' (BF16 batched dot-product extension){{$}}
vfwbdota.vv v8, v16, v12, 2, v0.t

# FP8 Batched Dot-Product Instructions

# CHECK-INST: vfqwbdota.vv v8, v16, v12, 1
# CHECK-ENCODING: [0x77,0x14,0x16,0xbb]
# CHECK-ERROR: instruction requires the following: 'Zvfqwbdota8f' (OCP FP8 batched dot-product extension){{$}}
vfqwbdota.vv v8, v16, v12, 1

# CHECK-INST: vfqwbdota.vv v8, v16, v12, 2, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xb9]
# CHECK-ERROR: instruction requires the following: 'Zvfqwbdota8f' (OCP FP8 batched dot-product extension){{$}}
vfqwbdota.vv v8, v16, v12, 2, v0.t

# CHECK-INST: vfqwbdota.alt.vv v8, v16, v12, 1
# CHECK-ENCODING: [0x77,0x14,0x16,0xbf]
# CHECK-ERROR: instruction requires the following: 'Zvfqwbdota8f' (OCP FP8 batched dot-product extension){{$}}
vfqwbdota.alt.vv v8, v16, v12, 1

# CHECK-INST: vfqwbdota.alt.vv v8, v16, v12, 2, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xbd]
# CHECK-ERROR: instruction requires the following: 'Zvfqwbdota8f' (OCP FP8 batched dot-product extension){{$}}
vfqwbdota.alt.vv v8, v16, v12, 2, v0.t

# FP32 Batched Dot-Product Instructions

# CHECK-INST: vfbdota.vv v8, v16, v12, 1
# CHECK-ENCODING: [0x77,0x14,0x16,0xaf]
# CHECK-ERROR: instruction requires the following: 'Zvfbdota32f' (FP32 batched dot-product extension){{$}}
vfbdota.vv v8, v16, v12, 1

# CHECK-INST: vfbdota.vv v8, v16, v12, 2, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xad]
# CHECK-ERROR: instruction requires the following: 'Zvfbdota32f' (FP32 batched dot-product extension){{$}}
vfbdota.vv v8, v16, v12, 2, v0.t
