# RUN: llvm-mc -triple=riscv32 --mattr=+xcvelw -show-encoding %s \
# RUN:     | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvelw < %s \
# RUN:     | llvm-objdump --mattr=+xcvelw --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --verify -triple riscv32 %s

cv.elw a0, 1024(a0) # expected-error {{instruction requires the following: 'XCVelw' (CORE-V Event Load Word)}}
# CHECK-INSTR: cv.elw a0, 1024(a0)
# CHECK-ENCODING: [0x0b,0x35,0x05,0x40]

cv.elw a1, 1(a1) # expected-error {{instruction requires the following: 'XCVelw' (CORE-V Event Load Word)}}
# CHECK-INSTR: cv.elw a1, 1(a1)
# CHECK-ENCODING: [0x8b,0xb5,0x15,0x00]

cv.elw a2, -1024(a3) # expected-error {{instruction requires the following: 'XCVelw' (CORE-V Event Load Word)}}
# CHECK-INSTR: cv.elw  a2, -1024(a3)
# CHECK-ENCODING: [0x0b,0xb6,0x06,0xc0]
