# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+v --mattr=+experimental-zvqwdota8i %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+v --mattr=+experimental-zvqwdota8i %s \
# RUN:        | llvm-objdump -d --mattr=+v --mattr=+experimental-zvqwdota8i --no-print-imm-hex  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vsetvli a2, a0, e8alt, m1, ta, ma # expected-error {{instruction requires the following: 'V' (Vector Extension for Application Processors), 'Zve32x' (Vector Extensions for Embedded Processors)}}
# CHECK-INST: vsetvli a2, a0, e8alt, m1, ta, ma
# CHECK-ENCODING: [0x57,0x76,0x05,0x1c]

vqwdotau.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvqwdota8i' (8-bit Integer Dot-Product) or 'Zvqwdota16i' (16-bit Integer Dot-Product)}}
# CHECK-INST: vqwdotau.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x05,0x94,0x9a]

vqwdotau.vv v10, v9, v8, v0.t # expected-error {{instruction requires the following: 'Zvqwdota8i' (8-bit Integer Dot-Product) or 'Zvqwdota16i' (16-bit Integer Dot-Product)}}
# CHECK-INST: vqwdotau.vv v10, v9, v8, v0.t
# CHECK-ENCODING: [0x77,0x05,0x94,0x98]
