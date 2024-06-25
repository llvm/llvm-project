# RUN: llvm-mc -triple=riscv32 -show-encoding --mattr=+zve32x --mattr=+zvksh %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INST
# RUN: not llvm-mc -triple=riscv32 -show-encoding %s 2>&1 \
# RUN:        | FileCheck %s --check-prefix=CHECK-ERROR
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvksh %s \
# RUN:        | llvm-objdump -d --mattr=+zve32x --mattr=+zvksh --no-print-imm-hex - \
# RUN:        | FileCheck %s --check-prefix=CHECK-INST
# RUN: llvm-mc -triple=riscv32 -filetype=obj --mattr=+zve32x --mattr=+zvksh %s \
# RUN:        | llvm-objdump -d - | FileCheck %s --check-prefix=CHECK-UNKNOWN

vsm3c.vi v10, v9, 7
# CHECK-INST: vsm3c.vi v10, v9, 7
# CHECK-ENCODING: [0x77,0xa5,0x93,0xae]
# CHECK-ERROR: instruction requires the following: 'Zvksh' (SM3 Hash Function Instructions){{$}}
# CHECK-UNKNOWN: ae93a577 <unknown>

vsm3me.vv v10, v9, v8
# CHECK-INST: vsm3me.vv v10, v9, v8
# CHECK-ENCODING: [0x77,0x25,0x94,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvksh' (SM3 Hash Function Instructions){{$}}
# CHECK-UNKNOWN: 82942577 <unknown>

# vs1 is allowed to overlap, but not vs2.
vsm3me.vv v10, v9, v10
# CHECK-INST: vsm3me.vv v10, v9, v10
# CHECK-ENCODING: [0x77,0x25,0x95,0x82]
# CHECK-ERROR: instruction requires the following: 'Zvksh' (SM3 Hash Function Instructions){{$}}
# CHECK-UNKNOWN: 82952577 <unknown>
