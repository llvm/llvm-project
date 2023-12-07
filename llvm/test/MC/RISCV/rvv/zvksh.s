# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+experimental-zvksh %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvksh %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+experimental-zvksh  - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+experimental-zvksh %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsm3c.vi v10, v9, 7
# CHECK-INST: vsm3c.vi v10, v9, 7
# CHECK-ENCODING: [0x77,0xa5,0x93,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvksh' (SM3 Hash Function Instructions){{$}}
# CHECK-UNKNOWN: 77 a5 93 ae   <unknown>

vsm3me.vv v10, v9, v8
# CHECK-INST: vsm3me.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvksh' (SM3 Hash Function Instructions){{$}}
# CHECK-UNKNOWN: 77 25 94 82   <unknown>
