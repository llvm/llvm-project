# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfwbdota16bf %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv64 -show-encoding -mattr=+v %s
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfwbdota16bf %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfwbdota16bf --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vsetvli zero, zero, e16alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x70,0x80,0x1c]
vsetvli zero, zero, e16alt, m1, ta, ma

# CHECK-INST: vfwbdota.vv v8, v16, v12, 8
# CHECK-ENCODING: [0x77,0x14,0x16,0xb3]
vfwbdota.vv v8, v16, v12, 8 # expected-error {{instruction requires the following: 'Zvfwbdota16bf' (BF16 batched dot-product extension)}}

# CHECK-INST: vfwbdota.vv v8, v16, v12, 16, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xb1]
vfwbdota.vv v8, v16, v12, 16, v0.t # expected-error {{instruction requires the following: 'Zvfwbdota16bf' (BF16 batched dot-product extension)}}
