# RUN: llvm-mc -triple=riscv64 -show-encoding -mattr=+experimental-zvfwbdota16bf %s \
# RUN:   | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv64 -show-encoding -mattr=+v %s 2>&1 \
# RUN:   | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv64 -filetype=obj -mattr=+experimental-zvfwbdota16bf %s \
# RUN:    | llvm-objdump -d --mattr=+experimental-zvfwbdota16bf --no-print-imm-hex - \
# RUN:    | FileCheck %s --check-prefix=CHECK-INST

# CHECK-INST: vsetvli zero, zero, e16alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x70,0x80,0x1c]
vsetvli zero, zero, e16alt, m1, ta, ma

# CHECK-INST: vfwbdota.vv v8, v16, v12, 1
# CHECK-ENCODING: [0x77,0x14,0x16,0xb3]
# CHECK-ERROR: instruction requires the following: 'Zvfwbdota16bf' (BF16 batched dot-product extension){{$}}
vfwbdota.vv v8, v16, v12, 1

# CHECK-INST: vfwbdota.vv v8, v16, v12, 2, v0.t
# CHECK-ENCODING: [0x77,0x14,0x26,0xb1]
# CHECK-ERROR: instruction requires the following: 'Zvfwbdota16bf' (BF16 batched dot-product extension){{$}}
vfwbdota.vv v8, v16, v12, 2, v0.t
