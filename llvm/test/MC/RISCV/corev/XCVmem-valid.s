# RUN: llvm-mc -triple=riscv32 --mattr=+xcvmem -show-encoding %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvmem < %s \
# RUN:     | llvm-objdump --no-print-imm-hex --mattr=+xcvmem -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --verify -triple riscv32 %s

cv.lb t0, (t1), 0 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lb t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x02,0x03,0x00]

cv.lb a0, (a1), 2047 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lb a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0x85,0xf5,0x7f]

cv.lb t0, (t1), t2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lb t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x00]

cv.lb a0, (a1), a2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lb a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x00]

cv.lb t0, t2(t1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lb t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x08]

cv.lb a0, a2(a1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lb a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x08]

cv.lbu t0, (t1), 0 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lbu t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x42,0x03,0x00]

cv.lbu a0, (a1), 2047 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lbu a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0xc5,0xf5,0x7f]

cv.lbu t0, (t1), t2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lbu t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x10]

cv.lbu a0, (a1), a2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lbu a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x10]

cv.lbu t0, t2(t1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lbu t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x18]

cv.lbu a0, a2(a1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lbu a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x18]

cv.lh t0, (t1), 0 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lh t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x12,0x03,0x00]

cv.lh a0, (a1), 2047 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lh a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0x95,0xf5,0x7f]

cv.lh t0, (t1), t2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lh t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x02]

cv.lh a0, (a1), a2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lh a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x02]

cv.lh t0, t2(t1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lh t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x0a]

cv.lh a0, a2(a1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lh a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x0a]

cv.lhu t0, (t1), 0 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lhu t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x52,0x03,0x00]

cv.lhu a0, (a1), 2047 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lhu a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0xd5,0xf5,0x7f]

cv.lhu t0, (t1), t2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lhu t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x12]

cv.lhu a0, (a1), a2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lhu a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x12]

cv.lhu t0, t2(t1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lhu t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x1a]

cv.lhu a0, a2(a1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lhu a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x1a]

cv.lw t0, (t1), 0 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lw t0, (t1), 0
# CHECK-ENCODING: [0x8b,0x22,0x03,0x00]

cv.lw a0, (a1), 2047 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lw a0, (a1), 2047
# CHECK-ENCODING: [0x0b,0xa5,0xf5,0x7f]

cv.lw t0, (t1), t2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lw t0, (t1), t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x04]

cv.lw a0, (a1), a2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lw a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x04]

cv.lw t0, t2(t1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lw t0, t2(t1)
# CHECK-ENCODING: [0xab,0x32,0x73,0x0c]

cv.lw a0, a2(a1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.lw a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x0c]

cv.sb t0, (t1), 0 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sb t0, (t1), 0
# CHECK-ENCODING: [0x2b,0x00,0x53,0x00]

cv.sb a0, (a1), 2047 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sb a0, (a1), 2047
# CHECK-ENCODING: [0xab,0x8f,0xa5,0x7e]

cv.sb t0, (t1), t2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sb t0, (t1), t2
# CHECK-ENCODING: [0xab,0x33,0x53,0x20]

cv.sb a0, (a1), a2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sb a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x20]

cv.sb t0, t2(t1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sb t0, t2(t1)
# CHECK-ENCODING: [0xab,0x33,0x53,0x28]

cv.sb a0, a2(a1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sb a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x28]

cv.sh t0, (t1), 0 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sh t0, (t1), 0
# CHECK-ENCODING: [0x2b,0x10,0x53,0x00]

cv.sh a0, (a1), 2047 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sh a0, (a1), 2047
# CHECK-ENCODING: [0xab,0x9f,0xa5,0x7e]

cv.sh t0, (t1), t2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sh t0, (t1), t2
# CHECK-ENCODING: [0xab,0x33,0x53,0x22]

cv.sh a0, (a1), a2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sh a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x22]

cv.sh t0, t2(t1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sh t0, t2(t1)
# CHECK-ENCODING: [0xab,0x33,0x53,0x2a]

cv.sh a0, a2(a1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sh a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x2a]

cv.sw t0, (t1), 0 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sw t0, (t1), 0
# CHECK-ENCODING: [0x2b,0x20,0x53,0x00]

cv.sw a0, (a1), 2047 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sw a0, (a1), 2047
# CHECK-ENCODING: [0xab,0xaf,0xa5,0x7e]

cv.sw t0, (t1), t2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sw t0, (t1), t2
# CHECK-ENCODING: [0xab,0x33,0x53,0x24]

cv.sw a0, (a1), a2 # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sw a0, (a1), a2
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x24]

cv.sw t0, t2(t1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sw t0, t2(t1)
# CHECK-ENCODING: [0xab,0x33,0x53,0x2c]

cv.sw a0, a2(a1) # expected-error {{instruction requires the following: 'XCVmem' (CORE-V Post-incrementing Load & Store)}}
# CHECK-INSTR: cv.sw a0, a2(a1)
# CHECK-ENCODING: [0x2b,0xb6,0xa5,0x2c]
