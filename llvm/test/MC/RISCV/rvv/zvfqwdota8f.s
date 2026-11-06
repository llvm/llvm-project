# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+v --mattr=+experimental-zvfqwdota8f %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v --mattr=+experimental-zvfqwdota8f %s \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+experimental-zvfqwdota8f --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vsetvli a2, a0, e8alt, m1, ta, ma
# CHECK-INST: vsetvli a2, a0, e8alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x1c]
# expected-error@-3 {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}

vfqwdota.vv v10, v9, v8
# CHECK-INST: vfqwdota.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x15,0x94,0x9a]
# expected-error@-3 {{instruction requires the following: 'Zvfqwdota8f' (OCP FP8 Dot-Product)}}

vfqwdota.vv v10, v9, v8, v0.t
# CHECK-INST: vfqwdota.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x77,0x15,0x94,0x98]
# expected-error@-3 {{instruction requires the following: 'Zvfqwdota8f' (OCP FP8 Dot-Product)}}

vfqwdota.alt.vv v10, v9, v8
# CHECK-INST: vfqwdota.alt.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x15,0x94,0x9e]
# expected-error@-3 {{instruction requires the following: 'Zvfqwdota8f' (OCP FP8 Dot-Product)}}

vfqwdota.alt.vv v10, v9, v8, v0.t
# CHECK-INST: vfqwdota.alt.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x77,0x15,0x94,0x9c]
# expected-error@-3 {{instruction requires the following: 'Zvfqwdota8f' (OCP FP8 Dot-Product)}}

