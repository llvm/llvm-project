# RUN: llvm-mc -triple=riscv32 --mattr=+xcvbitmanip -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvbitmanip < %s \
# RUN:     | llvm-objdump --mattr=+xcvbitmanip --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: llvm-mc --verify -triple riscv32 %s

cv.extract t0, t1, 0, 1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extract t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x02,0x13,0x00]

cv.extract a0, a1, 17, 18 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extract a0, a1, 17, 18
# CHECK-ENCODING: [0x5b,0x85,0x25,0x23]

cv.extract s0, s1, 30, 31 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extract s0, s1, 30, 31
# CHECK-ENCODING: [0x5b,0x84,0xf4,0x3d]

cv.extractu t0, t1, 0, 1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractu t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x02,0x13,0x40]

cv.extractu a0, a1, 17, 18 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractu a0, a1, 17, 18
# CHECK-ENCODING: [0x5b,0x85,0x25,0x63]

cv.extractu s0, s1, 30, 31 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractu s0, s1, 30, 31
# CHECK-ENCODING: [0x5b,0x84,0xf4,0x7d]

cv.insert t0, t1, 0, 1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.insert t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x02,0x13,0x80]

cv.insert a0, a1, 31, 0 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.insert a0, a1, 31, 0
# CHECK-ENCODING: [0x5b,0x85,0x05,0xbe]

cv.insert s0, s1, 0, 31 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.insert s0, s1, 0, 31
# CHECK-ENCODING: [0x5b,0x84,0xf4,0x81]

cv.bclr t0, t1, 0, 1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bclr t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x12,0x13,0x00]

cv.bclr a0, a1, 17, 18 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bclr a0, a1, 17, 18
# CHECK-ENCODING: [0x5b,0x95,0x25,0x23]

cv.bclr s0, s1, 30, 31 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bclr s0, s1, 30, 31
# CHECK-ENCODING: [0x5b,0x94,0xf4,0x3d]

cv.bset t0, t1, 0, 1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bset t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x12,0x13,0x40]

cv.bset a0, a1, 17, 18 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bset a0, a1, 17, 18
# CHECK-ENCODING: [0x5b,0x95,0x25,0x63]

cv.bset s0, s1, 30, 31 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bset s0, s1, 30, 31
# CHECK-ENCODING: [0x5b,0x94,0xf4,0x7d]

cv.bitrev t0, t1, 0, 1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bitrev t0, t1, 0, 1
# CHECK-ENCODING: [0xdb,0x12,0x13,0xc0]

cv.bitrev a0, a1, 1, 18 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bitrev a0, a1, 1, 18
# CHECK-ENCODING: [0x5b,0x95,0x25,0xc3]

cv.bitrev s0, s1, 2, 31 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bitrev s0, s1, 2, 31
# CHECK-ENCODING: [0x5b,0x94,0xf4,0xc5]

cv.extractr t0, t1, t2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x30]

cv.extractr a0, a1, a2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x30]

cv.extractr s0, s1, s2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractr s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x31]

cv.extractur t0, t1, t2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractur t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x32]

cv.extractur a0, a1, a2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractur a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x32]

cv.extractur s0, s1, s2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.extractur s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x33]

cv.insertr t0, t1, t2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.insertr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x34]

cv.insertr a0, a1, a2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.insertr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x34]

cv.insertr s0, s1, s2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.insertr s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x35]

cv.bclrr t0, t1, t2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bclrr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x38]

cv.bclrr a0, a1, a2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bclrr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x38]

cv.bclrr s0, s1, s2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bclrr s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x39]

cv.bsetr t0, t1, t2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bsetr t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x3a]

cv.bsetr a0, a1, a2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bsetr a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x3a]

cv.bsetr s0, s1, s2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.bsetr s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x3b]

cv.ror t0, t1, t2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.ror t0, t1, t2
# CHECK-ENCODING: [0xab,0x32,0x73,0x40]

cv.ror a0, a1, a2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.ror a0, a1, a2
# CHECK-ENCODING: [0x2b,0xb5,0xc5,0x40]

cv.ror s0, s1, s2 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.ror s0, s1, s2
# CHECK-ENCODING: [0x2b,0xb4,0x24,0x41]

cv.ff1 t0, t1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.ff1 t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x42]

cv.ff1 a0, a1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.ff1 a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x42]

cv.ff1 s0, s1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.ff1 s0, s1
# CHECK-ENCODING: [0x2b,0xb4,0x04,0x42]

cv.fl1 t0, t1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.fl1 t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x44]

cv.fl1 a0, a1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.fl1 a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x44]

cv.fl1 s0, s1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.fl1 s0, s1
# CHECK-ENCODING: [0x2b,0xb4,0x04,0x44]

cv.clb t0, t1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.clb t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x46]

cv.clb a0, a1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.clb a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x46]

cv.clb s0, s1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.clb s0, s1
# CHECK-ENCODING: [0x2b,0xb4,0x04,0x46]

cv.cnt t0, t1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.cnt t0, t1
# CHECK-ENCODING: [0xab,0x32,0x03,0x48]

cv.cnt a0, a1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.cnt a0, a1
# CHECK-ENCODING: [0x2b,0xb5,0x05,0x48]

cv.cnt s0, s1 # expected-error {{instruction requires the following: 'XCVbitmanip' (CORE-V Bit Manipulation)}}
# CHECK-INSTR: cv.cnt s0, s1
# CHECK-ENCODING: [0x2b,0xb4,0x04,0x48]

