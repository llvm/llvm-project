# RUN: llvm-mc -triple=riscv32 --mattr=+xcvbi -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvbi < %s \
# RUN:     | llvm-objdump --mattr=+xcvbi -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-OBJDUMP %s
# RUN: not llvm-mc -triple riscv32 %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

//===----------------------------------------------------------------------===//
// cv.beqimm
//===----------------------------------------------------------------------===//

label1:

cv.beqimm t0, 0, 0
# CHECK-INSTR: cv.beqimm t0, 0, 0
# CHECK-OBJDUMP: cv.beqimm t0, 0x0, 0x0 <label1>
# CHECK-ENCODING: [0x0b,0xe0,0x02,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVbi' (CORE-V Immediate Branching){{$}}

cv.beqimm a0, 5, 42
# CHECK-INSTR: cv.beqimm a0, 5, 42
# CHECK-OBJDUMP: cv.beqimm a0, 0x5, 0x2e <label2+0x22>
# CHECK-ENCODING: [0x0b,0x65,0x55,0x02]
# CHECK-NO-EXT: instruction requires the following: 'XCVbi' (CORE-V Immediate Branching){{$}}

cv.beqimm a0, -5, label1
# CHECK-INSTR: cv.beqimm a0, -5, label1
# CHECK-OBJDUMP: cv.beqimm a0, -0x5, 0x0 <label1>
# CHECK-ENCODING: [0x0b'A',0x60'A',0xb5'A',0x01'A']
# CHECK-ENCODING: fixup A - offset: 0, value: label1, kind: fixup_riscv_branch
# CHECK-NO-EXT: instruction requires the following: 'XCVbi' (CORE-V Immediate Branching){{$}}

//===----------------------------------------------------------------------===//
// cv.bneimm
//===----------------------------------------------------------------------===//

label2:

cv.bneimm t0, 0, 0
# CHECK-INSTR: cv.bneimm t0, 0, 0
# CHECK-OBJDUMP: cv.bneimm t0, 0x0, 0xc <label2>
# CHECK-ENCODING: [0x0b,0xf0,0x02,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVbi' (CORE-V Immediate Branching){{$}}

cv.bneimm a0, 5, 42
# CHECK-INSTR: cv.bneimm a0, 5, 42
# CHECK-OBJDUMP: cv.bneimm a0, 0x5, 0x3a <label2+0x2e>
# CHECK-ENCODING: [0x0b,0x75,0x55,0x02]
# CHECK-NO-EXT: instruction requires the following: 'XCVbi' (CORE-V Immediate Branching){{$}}

cv.bneimm a0, -5, label2
# CHECK-INSTR: cv.bneimm a0, -5, label2
# CHECK-OBJDUMP: cv.bneimm a0, -0x5, 0xc <label2>
# CHECK-ENCODING: [0x0b'A',0x70'A',0xb5'A',0x01'A']
# CHECK-ENCODING: fixup A - offset: 0, value: label2, kind: fixup_riscv_branch
# CHECK-NO-EXT: instruction requires the following: 'XCVbi' (CORE-V Immediate Branching){{$}}
