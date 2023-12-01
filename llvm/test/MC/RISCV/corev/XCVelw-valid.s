# RUN: llvm-mc -triple=riscv32 --mattr=+xcvelw -show-encoding %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvelw < %s \
# RUN:     | llvm-objdump --mattr=+xcvelw --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

cv.elw a0, 1024(a0)
# CHECK-INSTR: cv.elw a0, 1024(a0)
# CHECK-ENCODING: [0x0b,0x35,0x05,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVelw' (CORE-V Event Load Word){{$}} 

cv.elw a1, 1(a1)
# CHECK-INSTR: cv.elw a1, 1(a1)
# CHECK-ENCODING: [0x8b,0xb5,0x15,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVelw' (CORE-V Event Load Word){{$}}

cv.elw a2, -1024(a3)
# CHECK-INSTR: cv.elw  a2, -1024(a3)
# CHECK-ENCODING: [0x0b,0xb6,0x06,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVelw' (CORE-V Event Load Word){{$}}
