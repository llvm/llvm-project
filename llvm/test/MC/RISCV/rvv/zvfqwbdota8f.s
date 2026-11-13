# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfqwbdota8f %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding -mattr=+v %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfqwbdota8f %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfqwbdota8f --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vsetvli zero, zero, e8alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x70,0x00,0x1c]
vsetvli zero, zero, e8alt, m1, ta, ma

# CHECK-INST: vfqwbdota.vv v8, v16, v12, 8
# CHECK-ENCODING: [0x77,0x14,0x16,0xbb]
vfqwbdota.vv v8, v16, v12, 8 # expected-error {{instruction requires the following: 'Zvfqwbdota8f' (OCP FP8 batched dot-product extension)}}

# CHECK-INST: vfqwbdota.vv v8, v16, v12, 16, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xb9]
vfqwbdota.vv v8, v16, v12, 16, v0.t # expected-error {{instruction requires the following: 'Zvfqwbdota8f' (OCP FP8 batched dot-product extension)}}

# CHECK-INST: vfqwbdota.alt.vv v8, v16, v12, 8
# CHECK-ENCODING: [0x77,0x14,0x16,0xbf]
vfqwbdota.alt.vv v8, v16, v12, 8 # expected-error {{instruction requires the following: 'Zvfqwbdota8f' (OCP FP8 batched dot-product extension)}}

# CHECK-INST: vfqwbdota.alt.vv v8, v16, v12, 16, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xbd]
vfqwbdota.alt.vv v8, v16, v12, 16, v0.t # expected-error {{instruction requires the following: 'Zvfqwbdota8f' (OCP FP8 batched dot-product extension)}}
