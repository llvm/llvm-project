# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+v --mattr=+experimental-zvfwdota16bf %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v --mattr=+experimental-zvfwdota16bf %s \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+experimental-zvfwdota16bf --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vsetvli a2, a0, e16alt, m1, ta, ma
# CHECK-INST: vsetvli a2, a0, e16alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x85,0x1c]
# CHECK-ERROR: instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors){{$}}

vfwdota.vv v10, v9, v8
# CHECK-INST: vfwdota.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x15,0x94,0x92]
# CHECK-ERROR: instruction requires the following: 'Zvfwdota16bf' (BF16 Dot-Product)

vfwdota.vv v10, v9, v8, v0.t
# CHECK-INST: vfwdota.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x77,0x15,0x94,0x90]
# CHECK-ERROR: instruction requires the following: 'Zvfwdota16bf' (BF16 Dot-Product)
