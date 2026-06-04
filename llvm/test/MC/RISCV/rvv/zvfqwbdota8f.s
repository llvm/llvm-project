# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfqwbdota8f %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfqwbdota8f %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfqwbdota8f --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vsetvli zero, zero, e8alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x70,0x00,0x1c]
vsetvli zero, zero, e8alt, m1, ta, ma

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
