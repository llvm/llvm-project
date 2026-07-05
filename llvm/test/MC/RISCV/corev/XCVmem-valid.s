# RUN: llvm-mc -triple=riscv32 --mattr=+xcvmem -show-encoding %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvmem < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+xcvmem -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

cv.lb t0, (t1), 0
# CHECK-INSTR: cv.lb t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x02,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lb a0, (a1), 2047
# CHECK-INSTR: cv.lb a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0x85,0xf5,0x7f]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lb t0, (t1), t2
# CHECK-INSTR: cv.lb t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lb a0, (a1), a2
# CHECK-INSTR: cv.lb a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lb t0, t2(t1)
# CHECK-INSTR: cv.lb t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lb a0, a2(a1)
# CHECK-INSTR: cv.lb a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lbu t0, (t1), 0
# CHECK-INSTR: cv.lbu t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x42,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lbu a0, (a1), 2047
# CHECK-INSTR: cv.lbu a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0xc5,0xf5,0x7f]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lbu t0, (t1), t2
# CHECK-INSTR: cv.lbu t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lbu a0, (a1), a2
# CHECK-INSTR: cv.lbu a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lbu t0, t2(t1)
# CHECK-INSTR: cv.lbu t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lbu a0, a2(a1)
# CHECK-INSTR: cv.lbu a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lh t0, (t1), 0
# CHECK-INSTR: cv.lh t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x12,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lh a0, (a1), 2047
# CHECK-INSTR: cv.lh a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0x95,0xf5,0x7f]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lh t0, (t1), t2
# CHECK-INSTR: cv.lh t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x02]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lh a0, (a1), a2
# CHECK-INSTR: cv.lh a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x02]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lh t0, t2(t1)
# CHECK-INSTR: cv.lh t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x0a]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lh a0, a2(a1)
# CHECK-INSTR: cv.lh a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x0a]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lhu t0, (t1), 0
# CHECK-INSTR: cv.lhu t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x52,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lhu a0, (a1), 2047
# CHECK-INSTR: cv.lhu a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0xd5,0xf5,0x7f]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lhu t0, (t1), t2
# CHECK-INSTR: cv.lhu t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x12]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lhu a0, (a1), a2
# CHECK-INSTR: cv.lhu a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x12]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lhu t0, t2(t1)
# CHECK-INSTR: cv.lhu t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x1a]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lhu a0, a2(a1)
# CHECK-INSTR: cv.lhu a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x1a]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lw t0, (t1), 0
# CHECK-INSTR: cv.lw t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x22,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lw a0, (a1), 2047
# CHECK-INSTR: cv.lw a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0xa5,0xf5,0x7f]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lw t0, (t1), t2
# CHECK-INSTR: cv.lw t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lw a0, (a1), a2
# CHECK-INSTR: cv.lw a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lw t0, t2(t1)
# CHECK-INSTR: cv.lw t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.lw a0, a2(a1)
# CHECK-INSTR: cv.lw a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sb t0, (t1), 0
# CHECK-INSTR: cv.sb t0, (t1), 0
# CHECK-ENCODING: [0x2b,0x00,0x53,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sb a0, (a1), 2047
# CHECK-INSTR: cv.sb a0, (a1), 2047
# CHECK-ENCODING: [0xab,0x8f,0xa5,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sb t0, (t1), t2
# CHECK-INSTR: cv.sb t0, (t1), t2
# CHECK-ENCODING: [0xab,0x33,0x53,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sb a0, (a1), a2
# CHECK-INSTR: cv.sb a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sb t0, t2(t1)
# CHECK-INSTR: cv.sb t0, t2(t1)
# CHECK-ENCODING: [0xab,0x33,0x53,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sb a0, a2(a1)
# CHECK-INSTR: cv.sb a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sh t0, (t1), 0
# CHECK-INSTR: cv.sh t0, (t1), 0
# CHECK-ENCODING: [0x2b,0x10,0x53,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sh a0, (a1), 2047
# CHECK-INSTR: cv.sh a0, (a1), 2047
# CHECK-ENCODING: [0xab,0x9f,0xa5,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sh t0, (t1), t2
# CHECK-INSTR: cv.sh t0, (t1), t2
# CHECK-ENCODING: [0xab,0x33,0x53,0x22]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sh a0, (a1), a2
# CHECK-INSTR: cv.sh a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x22]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sh t0, t2(t1)
# CHECK-INSTR: cv.sh t0, t2(t1)
# CHECK-ENCODING: [0xab,0x33,0x53,0x2a]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sh a0, a2(a1)
# CHECK-INSTR: cv.sh a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x2a]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sw t0, (t1), 0
# CHECK-INSTR: cv.sw t0, (t1), 0
# CHECK-ENCODING: [0x2b,0x20,0x53,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sw a0, (a1), 2047
# CHECK-INSTR: cv.sw a0, (a1), 2047
# CHECK-ENCODING: [0xab,0xaf,0xa5,0x7e]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sw t0, (t1), t2
# CHECK-INSTR: cv.sw t0, (t1), t2
# CHECK-ENCODING: [0xab,0x33,0x53,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sw a0, (a1), a2
# CHECK-INSTR: cv.sw a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sw t0, t2(t1)
# CHECK-INSTR: cv.sw t0, t2(t1)
# CHECK-ENCODING: [0xab,0x33,0x53,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}

cv.sw a0, a2(a1)
# CHECK-INSTR: cv.sw a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store){{$}}
