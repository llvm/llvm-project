# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvksh %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: llvm-mc --verify -triple=riscv32 -show-encoding %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvksh %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvksh --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST

vsm3c.vi v10, v9, 7 # expected-error {{instruction requires the following: 'Zvksh' (SM3 Hash Function Instructions)}}
# CHECK-INST: vsm3c.vi v10, v9, 7
# CHECK-ENCODING: [0x77,0xa5,0x93,0xae]

vsm3me.vv v10, v9, v8 # expected-error {{instruction requires the following: 'Zvksh' (SM3 Hash Function Instructions)}}
# CHECK-INST: vsm3me.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0x82]

# vs1 is allowed to overlap, but not vs2.
vsm3me.vv v10, v9, v10 # expected-error {{instruction requires the following: 'Zvksh' (SM3 Hash Function Instructions)}}
# CHECK-INST: vsm3me.vv v10, v9, v10
# CHECK-ENCODING: [0x77,0x25,0x95,0x82]
