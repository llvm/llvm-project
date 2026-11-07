# RUN: llvm-mc -triple=riscv32 --mattr=+xcvbitmanip -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvbitmanip < %s \
# RUN:     | llvm-objdump --mattr=+xcvbitmanip --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --verify -triple riscv32 %s

cv.extract t0, t1, 0, 1
# CHECK-INSTR: cv.extract t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x02,0x13,0x00]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extract a0, a1, 17, 18
# CHECK-INSTR: cv.extract a0, a1, 17, 18
# CHECK-ENCODING: [0x5b,0x85,0x25,0x23]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extract s0, s1, 30, 31
# CHECK-INSTR: cv.extract s0, s1, 30, 31
# CHECK-ENCODING: [0x5b,0x84,0xf4,0x3d]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractu t0, t1, 0, 1
# CHECK-INSTR: cv.extractu t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x02,0x13,0x40]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractu a0, a1, 17, 18
# CHECK-INSTR: cv.extractu a0, a1, 17, 18
# CHECK-ENCODING: [0x5b,0x85,0x25,0x63]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractu s0, s1, 30, 31
# CHECK-INSTR: cv.extractu s0, s1, 30, 31
# CHECK-ENCODING: [0x5b,0x84,0xf4,0x7d]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.insert t0, t1, 0, 1
# CHECK-INSTR: cv.insert t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x02,0x13,0x80]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.insert a0, a1, 31, 0
# CHECK-INSTR: cv.insert a0, a1, 31, 0
# CHECK-ENCODING: [0x5b,0x85,0x05,0xbe]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.insert s0, s1, 0, 31
# CHECK-INSTR: cv.insert s0, s1, 0, 31
# CHECK-ENCODING: [0x5b,0x84,0xf4,0x81]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bclr t0, t1, 0, 1
# CHECK-INSTR: cv.bclr t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x12,0x13,0x00]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bclr a0, a1, 17, 18
# CHECK-INSTR: cv.bclr a0, a1, 17, 18
# CHECK-ENCODING: [0x5b,0x95,0x25,0x23]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bclr s0, s1, 30, 31
# CHECK-INSTR: cv.bclr s0, s1, 30, 31
# CHECK-ENCODING: [0x5b,0x94,0xf4,0x3d]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bset t0, t1, 0, 1
# CHECK-INSTR: cv.bset t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x12,0x13,0x40]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bset a0, a1, 17, 18
# CHECK-INSTR: cv.bset a0, a1, 17, 18
# CHECK-ENCODING: [0x5b,0x95,0x25,0x63]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bset s0, s1, 30, 31
# CHECK-INSTR: cv.bset s0, s1, 30, 31
# CHECK-ENCODING: [0x5b,0x94,0xf4,0x7d]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bitrev t0, t1, 0, 1
# CHECK-INSTR: cv.bitrev t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x12,0x13,0xc0]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bitrev a0, a1, 1, 18
# CHECK-INSTR: cv.bitrev a0, a1, 1, 18
# CHECK-ENCODING: [0x5b,0x95,0x25,0xc3]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bitrev s0, s1, 2, 31
# CHECK-INSTR: cv.bitrev s0, s1, 2, 31
# CHECK-ENCODING: [0x5b,0x94,0xf4,0xc5]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractr t0, t1, t2
# CHECK-INSTR: cv.extractr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x30]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractr a0, a1, a2
# CHECK-INSTR: cv.extractr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x30]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractr s0, s1, s2
# CHECK-INSTR: cv.extractr s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x31]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractur t0, t1, t2
# CHECK-INSTR: cv.extractur t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x32]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractur a0, a1, a2
# CHECK-INSTR: cv.extractur a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x32]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.extractur s0, s1, s2
# CHECK-INSTR: cv.extractur s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x33]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.insertr t0, t1, t2
# CHECK-INSTR: cv.insertr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x34]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.insertr a0, a1, a2
# CHECK-INSTR: cv.insertr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x34]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.insertr s0, s1, s2
# CHECK-INSTR: cv.insertr s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x35]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bclrr t0, t1, t2
# CHECK-INSTR: cv.bclrr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x38]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bclrr a0, a1, a2
# CHECK-INSTR: cv.bclrr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x38]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bclrr s0, s1, s2
# CHECK-INSTR: cv.bclrr s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x39]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bsetr t0, t1, t2
# CHECK-INSTR: cv.bsetr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x3a]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bsetr a0, a1, a2
# CHECK-INSTR: cv.bsetr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x3a]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.bsetr s0, s1, s2
# CHECK-INSTR: cv.bsetr s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x3b]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.ror t0, t1, t2
# CHECK-INSTR: cv.ror t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x40]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.ror a0, a1, a2
# CHECK-INSTR: cv.ror a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x40]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.ror s0, s1, s2
# CHECK-INSTR: cv.ror s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x41]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.ff1 t0, t1
# CHECK-INSTR: cv.ff1 t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x42]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.ff1 a0, a1
# CHECK-INSTR: cv.ff1 a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x42]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.ff1 s0, s1
# CHECK-INSTR: cv.ff1 s0, s1
# CHECK-ENCODING: [0x2b,0xb4,0x04,0x42]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.fl1 t0, t1
# CHECK-INSTR: cv.fl1 t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x44]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.fl1 a0, a1
# CHECK-INSTR: cv.fl1 a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x44]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.fl1 s0, s1
# CHECK-INSTR: cv.fl1 s0, s1
# CHECK-ENCODING: [0x2b,0xb4,0x04,0x44]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.clb t0, t1
# CHECK-INSTR: cv.clb t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x46]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.clb a0, a1
# CHECK-INSTR: cv.clb a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x46]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.clb s0, s1
# CHECK-INSTR: cv.clb s0, s1
# CHECK-ENCODING: [0x2b,0xb4,0x04,0x46]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.cnt t0, t1
# CHECK-INSTR: cv.cnt t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x48]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.cnt a0, a1
# CHECK-INSTR: cv.cnt a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x48]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

cv.cnt s0, s1
# CHECK-INSTR: cv.cnt s0, s1
# CHECK-ENCODING: [0x2b,0xb4,0x04,0x48]
# expected-error@-3 {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}

