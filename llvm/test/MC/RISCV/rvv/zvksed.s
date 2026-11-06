# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvksed %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvksed %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvksed --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vsm4k.vi v10, v9, 7
# CHECK-INST: vsm4k.vi v10, v9, 7
# CHECK-ENCODING: [0x77,0xa5,0x93,0x86]
# expected-error@-3 {{instruction requires the following: 'Zvksed' (SM4 Block Cipher Instructions)}}

vsm4k.vi v10, v9, 31
# CHECK-INST: vsm4k.vi v10, v9, 31
# CHECK-ENCODING: [0x77,0xa5,0x9f,0x86]
# expected-error@-3 {{instruction requires the following: 'Zvksed' (SM4 Block Cipher Instructions)}}

vsm4r.vv v10, v9
# CHECK-INST: vsm4r.vv v10, v9
# CHECK-ENCODING: [0x77,0x25,0x98,0xa2]
# expected-error@-3 {{instruction requires the following: 'Zvksed' (SM4 Block Cipher Instructions)}}

vsm4r.vs v10, v9
# CHECK-INST: vsm4r.vs v10, v9
# CHECK-ENCODING: [0x77,0x25,0x98,0xa6]
# expected-error@-3 {{instruction requires the following: 'Zvksed' (SM4 Block Cipher Instructions)}}
