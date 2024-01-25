# RUN: llvm-mc -triple=riscv32 --mattr=+xcvsimd -show-encoding %s \
# RUN:        | FileCheck %s --check-prefixes=CHECK-ENCODING,CHECK-INSTR
# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+xcvsimd < %s \
# RUN:     | llvm-objdump --mattr=+xcvsimd --no-print-imm-hex -M no-aliases -d -r - \
# RUN:     | FileCheck --check-prefix=CHECK-INSTR %s
# RUN: not llvm-mc -triple riscv32 %s 2>&1 \
# RUN:     | FileCheck -check-prefix=CHECK-NO-EXT %s

//===----------------------------------------------------------------------===//
// cv.add.h
//===----------------------------------------------------------------------===//

cv.add.h t0, t1, t2
# CHECK-INSTR: cv.add.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.h t3, t4, t5
# CHECK-INSTR: cv.add.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.h a0, a1, a2
# CHECK-INSTR: cv.add.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.h s0, s1, s2
# CHECK-INSTR: cv.add.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.add.b
//===----------------------------------------------------------------------===//

cv.add.b t0, t1, t2
# CHECK-INSTR: cv.add.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.b t3, t4, t5
# CHECK-INSTR: cv.add.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.b a0, a1, a2
# CHECK-INSTR: cv.add.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.b s0, s1, s2
# CHECK-INSTR: cv.add.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.add.sc.h
//===----------------------------------------------------------------------===//

cv.add.sc.h t0, t1, t2
# CHECK-INSTR: cv.add.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sc.h t3, t4, t5
# CHECK-INSTR: cv.add.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sc.h a0, a1, a2
# CHECK-INSTR: cv.add.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sc.h s0, s1, s2
# CHECK-INSTR: cv.add.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.add.sc.b
//===----------------------------------------------------------------------===//

cv.add.sc.b t0, t1, t2
# CHECK-INSTR: cv.add.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sc.b t3, t4, t5
# CHECK-INSTR: cv.add.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sc.b a0, a1, a2
# CHECK-INSTR: cv.add.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sc.b s0, s1, s2
# CHECK-INSTR: cv.add.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.add.sci.h
//===----------------------------------------------------------------------===//

cv.add.sci.h t0, t1, 0
# CHECK-INSTR: cv.add.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sci.h t3, t4, -32
# CHECK-INSTR: cv.add.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sci.h a0, a1, 7
# CHECK-INSTR: cv.add.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x02]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sci.h s0, s1, -1
# CHECK-INSTR: cv.add.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x03]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.add.sci.b
//===----------------------------------------------------------------------===//

cv.add.sci.b t0, t1, 0
# CHECK-INSTR: cv.add.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x00]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sci.b t3, t4, -32
# CHECK-INSTR: cv.add.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x01]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sci.b a0, a1, 7
# CHECK-INSTR: cv.add.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x02]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.sci.b s0, s1, -1
# CHECK-INSTR: cv.add.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x03]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.h
//===----------------------------------------------------------------------===//

cv.sub.h t0, t1, t2
# CHECK-INSTR: cv.sub.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.h t3, t4, t5
# CHECK-INSTR: cv.sub.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.h a0, a1, a2
# CHECK-INSTR: cv.sub.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.h s0, s1, s2
# CHECK-INSTR: cv.sub.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.b
//===----------------------------------------------------------------------===//

cv.sub.b t0, t1, t2
# CHECK-INSTR: cv.sub.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.b t3, t4, t5
# CHECK-INSTR: cv.sub.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.b a0, a1, a2
# CHECK-INSTR: cv.sub.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.b s0, s1, s2
# CHECK-INSTR: cv.sub.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.sc.h
//===----------------------------------------------------------------------===//

cv.sub.sc.h t0, t1, t2
# CHECK-INSTR: cv.sub.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sc.h t3, t4, t5
# CHECK-INSTR: cv.sub.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sc.h a0, a1, a2
# CHECK-INSTR: cv.sub.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sc.h s0, s1, s2
# CHECK-INSTR: cv.sub.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.sc.b
//===----------------------------------------------------------------------===//

cv.sub.sc.b t0, t1, t2
# CHECK-INSTR: cv.sub.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sc.b t3, t4, t5
# CHECK-INSTR: cv.sub.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sc.b a0, a1, a2
# CHECK-INSTR: cv.sub.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sc.b s0, s1, s2
# CHECK-INSTR: cv.sub.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.sci.h
//===----------------------------------------------------------------------===//

cv.sub.sci.h t0, t1, 0
# CHECK-INSTR: cv.sub.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sci.h t3, t4, -32
# CHECK-INSTR: cv.sub.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sci.h a0, a1, 7
# CHECK-INSTR: cv.sub.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x0a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sci.h s0, s1, -1
# CHECK-INSTR: cv.sub.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x0b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.sci.b
//===----------------------------------------------------------------------===//

cv.sub.sci.b t0, t1, 0
# CHECK-INSTR: cv.sub.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x08]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sci.b t3, t4, -32
# CHECK-INSTR: cv.sub.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x09]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sci.b a0, a1, 7
# CHECK-INSTR: cv.sub.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x0a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.sci.b s0, s1, -1
# CHECK-INSTR: cv.sub.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x0b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avg.h
//===----------------------------------------------------------------------===//

cv.avg.h t0, t1, t2
# CHECK-INSTR: cv.avg.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.h t3, t4, t5
# CHECK-INSTR: cv.avg.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.h a0, a1, a2
# CHECK-INSTR: cv.avg.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.h s0, s1, s2
# CHECK-INSTR: cv.avg.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avg.b
//===----------------------------------------------------------------------===//

cv.avg.b t0, t1, t2
# CHECK-INSTR: cv.avg.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.b t3, t4, t5
# CHECK-INSTR: cv.avg.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.b a0, a1, a2
# CHECK-INSTR: cv.avg.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.b s0, s1, s2
# CHECK-INSTR: cv.avg.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avg.sc.h
//===----------------------------------------------------------------------===//

cv.avg.sc.h t0, t1, t2
# CHECK-INSTR: cv.avg.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sc.h t3, t4, t5
# CHECK-INSTR: cv.avg.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sc.h a0, a1, a2
# CHECK-INSTR: cv.avg.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sc.h s0, s1, s2
# CHECK-INSTR: cv.avg.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avg.sc.b
//===----------------------------------------------------------------------===//

cv.avg.sc.b t0, t1, t2
# CHECK-INSTR: cv.avg.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sc.b t3, t4, t5
# CHECK-INSTR: cv.avg.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sc.b a0, a1, a2
# CHECK-INSTR: cv.avg.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sc.b s0, s1, s2
# CHECK-INSTR: cv.avg.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avg.sci.h
//===----------------------------------------------------------------------===//

cv.avg.sci.h t0, t1, 0
# CHECK-INSTR: cv.avg.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sci.h t3, t4, -32
# CHECK-INSTR: cv.avg.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sci.h a0, a1, 7
# CHECK-INSTR: cv.avg.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x12]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sci.h s0, s1, -1
# CHECK-INSTR: cv.avg.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x13]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avg.sci.b
//===----------------------------------------------------------------------===//

cv.avg.sci.b t0, t1, 0
# CHECK-INSTR: cv.avg.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x10]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sci.b t3, t4, -32
# CHECK-INSTR: cv.avg.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x11]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sci.b a0, a1, 7
# CHECK-INSTR: cv.avg.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x12]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avg.sci.b s0, s1, -1
# CHECK-INSTR: cv.avg.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x13]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avgu.h
//===----------------------------------------------------------------------===//

cv.avgu.h t0, t1, t2
# CHECK-INSTR: cv.avgu.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.h t3, t4, t5
# CHECK-INSTR: cv.avgu.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.h a0, a1, a2
# CHECK-INSTR: cv.avgu.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.h s0, s1, s2
# CHECK-INSTR: cv.avgu.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avgu.b
//===----------------------------------------------------------------------===//

cv.avgu.b t0, t1, t2
# CHECK-INSTR: cv.avgu.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.b t3, t4, t5
# CHECK-INSTR: cv.avgu.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.b a0, a1, a2
# CHECK-INSTR: cv.avgu.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.b s0, s1, s2
# CHECK-INSTR: cv.avgu.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avgu.sc.h
//===----------------------------------------------------------------------===//

cv.avgu.sc.h t0, t1, t2
# CHECK-INSTR: cv.avgu.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sc.h t3, t4, t5
# CHECK-INSTR: cv.avgu.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sc.h a0, a1, a2
# CHECK-INSTR: cv.avgu.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sc.h s0, s1, s2
# CHECK-INSTR: cv.avgu.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avgu.sc.b
//===----------------------------------------------------------------------===//

cv.avgu.sc.b t0, t1, t2
# CHECK-INSTR: cv.avgu.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sc.b t3, t4, t5
# CHECK-INSTR: cv.avgu.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sc.b a0, a1, a2
# CHECK-INSTR: cv.avgu.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sc.b s0, s1, s2
# CHECK-INSTR: cv.avgu.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avgu.sci.h
//===----------------------------------------------------------------------===//

cv.avgu.sci.h t0, t1, 0
# CHECK-INSTR: cv.avgu.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sci.h t3, t4, 32
# CHECK-INSTR: cv.avgu.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sci.h a0, a1, 7
# CHECK-INSTR: cv.avgu.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x1a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sci.h s0, s1, 63
# CHECK-INSTR: cv.avgu.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x1b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.avgu.sci.b
//===----------------------------------------------------------------------===//

cv.avgu.sci.b t0, t1, 0
# CHECK-INSTR: cv.avgu.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x18]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sci.b t3, t4, 32
# CHECK-INSTR: cv.avgu.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x19]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sci.b a0, a1, 7
# CHECK-INSTR: cv.avgu.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x1a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.avgu.sci.b s0, s1, 63
# CHECK-INSTR: cv.avgu.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x1b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.min.h
//===----------------------------------------------------------------------===//

cv.min.h t0, t1, t2
# CHECK-INSTR: cv.min.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.h t3, t4, t5
# CHECK-INSTR: cv.min.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.h a0, a1, a2
# CHECK-INSTR: cv.min.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.h s0, s1, s2
# CHECK-INSTR: cv.min.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.min.b
//===----------------------------------------------------------------------===//

cv.min.b t0, t1, t2
# CHECK-INSTR: cv.min.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.b t3, t4, t5
# CHECK-INSTR: cv.min.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.b a0, a1, a2
# CHECK-INSTR: cv.min.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.b s0, s1, s2
# CHECK-INSTR: cv.min.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.min.sc.h
//===----------------------------------------------------------------------===//

cv.min.sc.h t0, t1, t2
# CHECK-INSTR: cv.min.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sc.h t3, t4, t5
# CHECK-INSTR: cv.min.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sc.h a0, a1, a2
# CHECK-INSTR: cv.min.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sc.h s0, s1, s2
# CHECK-INSTR: cv.min.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.min.sc.b
//===----------------------------------------------------------------------===//

cv.min.sc.b t0, t1, t2
# CHECK-INSTR: cv.min.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sc.b t3, t4, t5
# CHECK-INSTR: cv.min.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sc.b a0, a1, a2
# CHECK-INSTR: cv.min.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sc.b s0, s1, s2
# CHECK-INSTR: cv.min.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.min.sci.h
//===----------------------------------------------------------------------===//

cv.min.sci.h t0, t1, 0
# CHECK-INSTR: cv.min.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sci.h t3, t4, -32
# CHECK-INSTR: cv.min.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sci.h a0, a1, 7
# CHECK-INSTR: cv.min.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x22]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sci.h s0, s1, -1
# CHECK-INSTR: cv.min.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x23]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.min.sci.b
//===----------------------------------------------------------------------===//

cv.min.sci.b t0, t1, 0
# CHECK-INSTR: cv.min.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x20]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sci.b t3, t4, -32
# CHECK-INSTR: cv.min.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x21]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sci.b a0, a1, 7
# CHECK-INSTR: cv.min.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x22]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.min.sci.b s0, s1, -1
# CHECK-INSTR: cv.min.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x23]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.minu.h
//===----------------------------------------------------------------------===//

cv.minu.h t0, t1, t2
# CHECK-INSTR: cv.minu.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.h t3, t4, t5
# CHECK-INSTR: cv.minu.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.h a0, a1, a2
# CHECK-INSTR: cv.minu.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.h s0, s1, s2
# CHECK-INSTR: cv.minu.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.minu.b
//===----------------------------------------------------------------------===//

cv.minu.b t0, t1, t2
# CHECK-INSTR: cv.minu.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.b t3, t4, t5
# CHECK-INSTR: cv.minu.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.b a0, a1, a2
# CHECK-INSTR: cv.minu.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.b s0, s1, s2
# CHECK-INSTR: cv.minu.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.minu.sc.h
//===----------------------------------------------------------------------===//

cv.minu.sc.h t0, t1, t2
# CHECK-INSTR: cv.minu.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sc.h t3, t4, t5
# CHECK-INSTR: cv.minu.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sc.h a0, a1, a2
# CHECK-INSTR: cv.minu.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sc.h s0, s1, s2
# CHECK-INSTR: cv.minu.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.minu.sc.b
//===----------------------------------------------------------------------===//

cv.minu.sc.b t0, t1, t2
# CHECK-INSTR: cv.minu.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sc.b t3, t4, t5
# CHECK-INSTR: cv.minu.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sc.b a0, a1, a2
# CHECK-INSTR: cv.minu.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sc.b s0, s1, s2
# CHECK-INSTR: cv.minu.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.minu.sci.h
//===----------------------------------------------------------------------===//

cv.minu.sci.h t0, t1, 0
# CHECK-INSTR: cv.minu.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sci.h t3, t4, 32
# CHECK-INSTR: cv.minu.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sci.h a0, a1, 7
# CHECK-INSTR: cv.minu.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x2a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sci.h s0, s1, 63
# CHECK-INSTR: cv.minu.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x2b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.minu.sci.b
//===----------------------------------------------------------------------===//

cv.minu.sci.b t0, t1, 0
# CHECK-INSTR: cv.minu.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x28]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sci.b t3, t4, 32
# CHECK-INSTR: cv.minu.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x29]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sci.b a0, a1, 7
# CHECK-INSTR: cv.minu.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x2a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.minu.sci.b s0, s1, 63
# CHECK-INSTR: cv.minu.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x2b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.max.h
//===----------------------------------------------------------------------===//

cv.max.h t0, t1, t2
# CHECK-INSTR: cv.max.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.h t3, t4, t5
# CHECK-INSTR: cv.max.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.h a0, a1, a2
# CHECK-INSTR: cv.max.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.h s0, s1, s2
# CHECK-INSTR: cv.max.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.max.b
//===----------------------------------------------------------------------===//

cv.max.b t0, t1, t2
# CHECK-INSTR: cv.max.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.b t3, t4, t5
# CHECK-INSTR: cv.max.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.b a0, a1, a2
# CHECK-INSTR: cv.max.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.b s0, s1, s2
# CHECK-INSTR: cv.max.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.max.sc.h
//===----------------------------------------------------------------------===//

cv.max.sc.h t0, t1, t2
# CHECK-INSTR: cv.max.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sc.h t3, t4, t5
# CHECK-INSTR: cv.max.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sc.h a0, a1, a2
# CHECK-INSTR: cv.max.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sc.h s0, s1, s2
# CHECK-INSTR: cv.max.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.max.sc.b
//===----------------------------------------------------------------------===//

cv.max.sc.b t0, t1, t2
# CHECK-INSTR: cv.max.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sc.b t3, t4, t5
# CHECK-INSTR: cv.max.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sc.b a0, a1, a2
# CHECK-INSTR: cv.max.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sc.b s0, s1, s2
# CHECK-INSTR: cv.max.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.max.sci.h
//===----------------------------------------------------------------------===//

cv.max.sci.h t0, t1, 0
# CHECK-INSTR: cv.max.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sci.h t3, t4, -32
# CHECK-INSTR: cv.max.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sci.h a0, a1, 7
# CHECK-INSTR: cv.max.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x32]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sci.h s0, s1, -1
# CHECK-INSTR: cv.max.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x33]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.max.sci.b
//===----------------------------------------------------------------------===//

cv.max.sci.b t0, t1, 0
# CHECK-INSTR: cv.max.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x30]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sci.b t3, t4, -32
# CHECK-INSTR: cv.max.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x31]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sci.b a0, a1, 7
# CHECK-INSTR: cv.max.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x32]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.max.sci.b s0, s1, -1
# CHECK-INSTR: cv.max.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x33]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.maxu.h
//===----------------------------------------------------------------------===//

cv.maxu.h t0, t1, t2
# CHECK-INSTR: cv.maxu.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.h t3, t4, t5
# CHECK-INSTR: cv.maxu.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.h a0, a1, a2
# CHECK-INSTR: cv.maxu.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.h s0, s1, s2
# CHECK-INSTR: cv.maxu.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.maxu.b
//===----------------------------------------------------------------------===//

cv.maxu.b t0, t1, t2
# CHECK-INSTR: cv.maxu.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.b t3, t4, t5
# CHECK-INSTR: cv.maxu.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.b a0, a1, a2
# CHECK-INSTR: cv.maxu.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.b s0, s1, s2
# CHECK-INSTR: cv.maxu.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.maxu.sc.h
//===----------------------------------------------------------------------===//

cv.maxu.sc.h t0, t1, t2
# CHECK-INSTR: cv.maxu.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sc.h t3, t4, t5
# CHECK-INSTR: cv.maxu.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sc.h a0, a1, a2
# CHECK-INSTR: cv.maxu.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sc.h s0, s1, s2
# CHECK-INSTR: cv.maxu.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.maxu.sc.b
//===----------------------------------------------------------------------===//

cv.maxu.sc.b t0, t1, t2
# CHECK-INSTR: cv.maxu.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sc.b t3, t4, t5
# CHECK-INSTR: cv.maxu.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sc.b a0, a1, a2
# CHECK-INSTR: cv.maxu.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sc.b s0, s1, s2
# CHECK-INSTR: cv.maxu.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.maxu.sci.h
//===----------------------------------------------------------------------===//

cv.maxu.sci.h t0, t1, 0
# CHECK-INSTR: cv.maxu.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sci.h t3, t4, 32
# CHECK-INSTR: cv.maxu.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sci.h a0, a1, 7
# CHECK-INSTR: cv.maxu.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x3a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sci.h s0, s1, 63
# CHECK-INSTR: cv.maxu.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x3b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.maxu.sci.b
//===----------------------------------------------------------------------===//

cv.maxu.sci.b t0, t1, 0
# CHECK-INSTR: cv.maxu.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x38]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sci.b t3, t4, 32
# CHECK-INSTR: cv.maxu.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x39]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sci.b a0, a1, 7
# CHECK-INSTR: cv.maxu.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x3a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.maxu.sci.b s0, s1, 63
# CHECK-INSTR: cv.maxu.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x3b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.srl.h
//===----------------------------------------------------------------------===//

cv.srl.h t0, t1, t2
# CHECK-INSTR: cv.srl.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.h t3, t4, t5
# CHECK-INSTR: cv.srl.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x41]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.h a0, a1, a2
# CHECK-INSTR: cv.srl.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.h s0, s1, s2
# CHECK-INSTR: cv.srl.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x41]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.srl.b
//===----------------------------------------------------------------------===//

cv.srl.b t0, t1, t2
# CHECK-INSTR: cv.srl.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.b t3, t4, t5
# CHECK-INSTR: cv.srl.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x41]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.b a0, a1, a2
# CHECK-INSTR: cv.srl.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.b s0, s1, s2
# CHECK-INSTR: cv.srl.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x41]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.srl.sc.h
//===----------------------------------------------------------------------===//

cv.srl.sc.h t0, t1, t2
# CHECK-INSTR: cv.srl.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sc.h t3, t4, t5
# CHECK-INSTR: cv.srl.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x41]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sc.h a0, a1, a2
# CHECK-INSTR: cv.srl.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sc.h s0, s1, s2
# CHECK-INSTR: cv.srl.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x41]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.srl.sc.b
//===----------------------------------------------------------------------===//

cv.srl.sc.b t0, t1, t2
# CHECK-INSTR: cv.srl.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sc.b t3, t4, t5
# CHECK-INSTR: cv.srl.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x41]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sc.b a0, a1, a2
# CHECK-INSTR: cv.srl.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sc.b s0, s1, s2
# CHECK-INSTR: cv.srl.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x41]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.srl.sci.h
//===----------------------------------------------------------------------===//

cv.srl.sci.h t0, t1, 0
# CHECK-INSTR: cv.srl.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sci.h t3, t4, 0
# CHECK-INSTR: cv.srl.sci.h t3, t4, 0
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sci.h a0, a1, 7
# CHECK-INSTR: cv.srl.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x42]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sci.h s0, s1, 15
# CHECK-INSTR: cv.srl.sci.h s0, s1, 15
# CHECK-ENCODING: [0x7b,0xe4,0x74,0x42]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.srl.sci.b
//===----------------------------------------------------------------------===//

cv.srl.sci.b t0, t1, 0
# CHECK-INSTR: cv.srl.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sci.b t3, t4, 0
# CHECK-INSTR: cv.srl.sci.b t3, t4, 0
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x40]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sci.b a0, a1, 7
# CHECK-INSTR: cv.srl.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x42]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.srl.sci.b s0, s1, 7
# CHECK-INSTR: cv.srl.sci.b s0, s1, 7
# CHECK-ENCODING: [0x7b,0xf4,0x34,0x42]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sra.h
//===----------------------------------------------------------------------===//

cv.sra.h t0, t1, t2
# CHECK-INSTR: cv.sra.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.h t3, t4, t5
# CHECK-INSTR: cv.sra.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x49]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.h a0, a1, a2
# CHECK-INSTR: cv.sra.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.h s0, s1, s2
# CHECK-INSTR: cv.sra.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x49]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sra.b
//===----------------------------------------------------------------------===//

cv.sra.b t0, t1, t2
# CHECK-INSTR: cv.sra.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.b t3, t4, t5
# CHECK-INSTR: cv.sra.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x49]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.b a0, a1, a2
# CHECK-INSTR: cv.sra.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.b s0, s1, s2
# CHECK-INSTR: cv.sra.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x49]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sra.sc.h
//===----------------------------------------------------------------------===//

cv.sra.sc.h t0, t1, t2
# CHECK-INSTR: cv.sra.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sc.h t3, t4, t5
# CHECK-INSTR: cv.sra.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x49]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sc.h a0, a1, a2
# CHECK-INSTR: cv.sra.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sc.h s0, s1, s2
# CHECK-INSTR: cv.sra.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x49]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sra.sc.b
//===----------------------------------------------------------------------===//

cv.sra.sc.b t0, t1, t2
# CHECK-INSTR: cv.sra.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sc.b t3, t4, t5
# CHECK-INSTR: cv.sra.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x49]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sc.b a0, a1, a2
# CHECK-INSTR: cv.sra.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sc.b s0, s1, s2
# CHECK-INSTR: cv.sra.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x49]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sra.sci.h
//===----------------------------------------------------------------------===//

cv.sra.sci.h t0, t1, 0
# CHECK-INSTR: cv.sra.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sci.h t3, t4, 0
# CHECK-INSTR: cv.sra.sci.h t3, t4, 0
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sci.h a0, a1, 7
# CHECK-INSTR: cv.sra.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x4a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sci.h s0, s1, 15
# CHECK-INSTR: cv.sra.sci.h s0, s1, 15
# CHECK-ENCODING: [0x7b,0xe4,0x74,0x4a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sra.sci.b
//===----------------------------------------------------------------------===//

cv.sra.sci.b t0, t1, 0
# CHECK-INSTR: cv.sra.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sci.b t3, t4, 0
# CHECK-INSTR: cv.sra.sci.b t3, t4, 0
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x48]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sci.b a0, a1, 7
# CHECK-INSTR: cv.sra.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x4a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sra.sci.b s0, s1, 7
# CHECK-INSTR: cv.sra.sci.b s0, s1, 7
# CHECK-ENCODING: [0x7b,0xf4,0x34,0x4a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sll.h
//===----------------------------------------------------------------------===//

cv.sll.h t0, t1, t2
# CHECK-INSTR: cv.sll.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.h t3, t4, t5
# CHECK-INSTR: cv.sll.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x51]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.h a0, a1, a2
# CHECK-INSTR: cv.sll.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.h s0, s1, s2
# CHECK-INSTR: cv.sll.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x51]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sll.b
//===----------------------------------------------------------------------===//

cv.sll.b t0, t1, t2
# CHECK-INSTR: cv.sll.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.b t3, t4, t5
# CHECK-INSTR: cv.sll.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x51]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.b a0, a1, a2
# CHECK-INSTR: cv.sll.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.b s0, s1, s2
# CHECK-INSTR: cv.sll.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x51]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sll.sc.h
//===----------------------------------------------------------------------===//

cv.sll.sc.h t0, t1, t2
# CHECK-INSTR: cv.sll.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sc.h t3, t4, t5
# CHECK-INSTR: cv.sll.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x51]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sc.h a0, a1, a2
# CHECK-INSTR: cv.sll.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sc.h s0, s1, s2
# CHECK-INSTR: cv.sll.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x51]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sll.sc.b
//===----------------------------------------------------------------------===//

cv.sll.sc.b t0, t1, t2
# CHECK-INSTR: cv.sll.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sc.b t3, t4, t5
# CHECK-INSTR: cv.sll.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x51]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sc.b a0, a1, a2
# CHECK-INSTR: cv.sll.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sc.b s0, s1, s2
# CHECK-INSTR: cv.sll.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x51]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sll.sci.h
//===----------------------------------------------------------------------===//

cv.sll.sci.h t0, t1, 0
# CHECK-INSTR: cv.sll.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sci.h t3, t4, 0
# CHECK-INSTR: cv.sll.sci.h t3, t4, 0
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sci.h a0, a1, 7
# CHECK-INSTR: cv.sll.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x52]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sci.h s0, s1, 15
# CHECK-INSTR: cv.sll.sci.h s0, s1, 15
# CHECK-ENCODING: [0x7b,0xe4,0x74,0x52]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sll.sci.b
//===----------------------------------------------------------------------===//

cv.sll.sci.b t0, t1, 0
# CHECK-INSTR: cv.sll.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sci.b t3, t4, 0
# CHECK-INSTR: cv.sll.sci.b t3, t4, 0
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x50]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sci.b a0, a1, 7
# CHECK-INSTR: cv.sll.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x52]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sll.sci.b s0, s1, 7
# CHECK-INSTR: cv.sll.sci.b s0, s1, 7
# CHECK-ENCODING: [0x7b,0xf4,0x34,0x52]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.or.h
//===----------------------------------------------------------------------===//

cv.or.h t0, t1, t2
# CHECK-INSTR: cv.or.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.h t3, t4, t5
# CHECK-INSTR: cv.or.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.h a0, a1, a2
# CHECK-INSTR: cv.or.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.h s0, s1, s2
# CHECK-INSTR: cv.or.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.or.b
//===----------------------------------------------------------------------===//

cv.or.b t0, t1, t2
# CHECK-INSTR: cv.or.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.b t3, t4, t5
# CHECK-INSTR: cv.or.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.b a0, a1, a2
# CHECK-INSTR: cv.or.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.b s0, s1, s2
# CHECK-INSTR: cv.or.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.or.sc.h
//===----------------------------------------------------------------------===//

cv.or.sc.h t0, t1, t2
# CHECK-INSTR: cv.or.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sc.h t3, t4, t5
# CHECK-INSTR: cv.or.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sc.h a0, a1, a2
# CHECK-INSTR: cv.or.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sc.h s0, s1, s2
# CHECK-INSTR: cv.or.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.or.sc.b
//===----------------------------------------------------------------------===//

cv.or.sc.b t0, t1, t2
# CHECK-INSTR: cv.or.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sc.b t3, t4, t5
# CHECK-INSTR: cv.or.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sc.b a0, a1, a2
# CHECK-INSTR: cv.or.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sc.b s0, s1, s2
# CHECK-INSTR: cv.or.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.or.sci.h
//===----------------------------------------------------------------------===//

cv.or.sci.h t0, t1, 0
# CHECK-INSTR: cv.or.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sci.h t3, t4, -32
# CHECK-INSTR: cv.or.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sci.h a0, a1, 7
# CHECK-INSTR: cv.or.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x5a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sci.h s0, s1, -1
# CHECK-INSTR: cv.or.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x5b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.or.sci.b
//===----------------------------------------------------------------------===//

cv.or.sci.b t0, t1, 0
# CHECK-INSTR: cv.or.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x58]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sci.b t3, t4, -32
# CHECK-INSTR: cv.or.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x59]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sci.b a0, a1, 7
# CHECK-INSTR: cv.or.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x5a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.or.sci.b s0, s1, -1
# CHECK-INSTR: cv.or.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x5b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.xor.h
//===----------------------------------------------------------------------===//

cv.xor.h t0, t1, t2
# CHECK-INSTR: cv.xor.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.h t3, t4, t5
# CHECK-INSTR: cv.xor.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.h a0, a1, a2
# CHECK-INSTR: cv.xor.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.h s0, s1, s2
# CHECK-INSTR: cv.xor.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.xor.b
//===----------------------------------------------------------------------===//

cv.xor.b t0, t1, t2
# CHECK-INSTR: cv.xor.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.b t3, t4, t5
# CHECK-INSTR: cv.xor.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.b a0, a1, a2
# CHECK-INSTR: cv.xor.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.b s0, s1, s2
# CHECK-INSTR: cv.xor.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.xor.sc.h
//===----------------------------------------------------------------------===//

cv.xor.sc.h t0, t1, t2
# CHECK-INSTR: cv.xor.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sc.h t3, t4, t5
# CHECK-INSTR: cv.xor.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sc.h a0, a1, a2
# CHECK-INSTR: cv.xor.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sc.h s0, s1, s2
# CHECK-INSTR: cv.xor.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.xor.sc.b
//===----------------------------------------------------------------------===//

cv.xor.sc.b t0, t1, t2
# CHECK-INSTR: cv.xor.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sc.b t3, t4, t5
# CHECK-INSTR: cv.xor.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sc.b a0, a1, a2
# CHECK-INSTR: cv.xor.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sc.b s0, s1, s2
# CHECK-INSTR: cv.xor.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.xor.sci.h
//===----------------------------------------------------------------------===//

cv.xor.sci.h t0, t1, 0
# CHECK-INSTR: cv.xor.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sci.h t3, t4, -32
# CHECK-INSTR: cv.xor.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sci.h a0, a1, 7
# CHECK-INSTR: cv.xor.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x62]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sci.h s0, s1, -1
# CHECK-INSTR: cv.xor.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x63]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.xor.sci.b
//===----------------------------------------------------------------------===//

cv.xor.sci.b t0, t1, 0
# CHECK-INSTR: cv.xor.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x60]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sci.b t3, t4, -32
# CHECK-INSTR: cv.xor.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x61]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sci.b a0, a1, 7
# CHECK-INSTR: cv.xor.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x62]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.xor.sci.b s0, s1, -1
# CHECK-INSTR: cv.xor.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x63]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.and.h
//===----------------------------------------------------------------------===//

cv.and.h t0, t1, t2
# CHECK-INSTR: cv.and.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.h t3, t4, t5
# CHECK-INSTR: cv.and.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.h a0, a1, a2
# CHECK-INSTR: cv.and.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.h s0, s1, s2
# CHECK-INSTR: cv.and.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.and.b
//===----------------------------------------------------------------------===//

cv.and.b t0, t1, t2
# CHECK-INSTR: cv.and.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.b t3, t4, t5
# CHECK-INSTR: cv.and.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.b a0, a1, a2
# CHECK-INSTR: cv.and.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.b s0, s1, s2
# CHECK-INSTR: cv.and.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.and.sc.h
//===----------------------------------------------------------------------===//

cv.and.sc.h t0, t1, t2
# CHECK-INSTR: cv.and.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sc.h t3, t4, t5
# CHECK-INSTR: cv.and.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sc.h a0, a1, a2
# CHECK-INSTR: cv.and.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sc.h s0, s1, s2
# CHECK-INSTR: cv.and.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.and.sc.b
//===----------------------------------------------------------------------===//

cv.and.sc.b t0, t1, t2
# CHECK-INSTR: cv.and.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sc.b t3, t4, t5
# CHECK-INSTR: cv.and.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sc.b a0, a1, a2
# CHECK-INSTR: cv.and.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sc.b s0, s1, s2
# CHECK-INSTR: cv.and.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.and.sci.h
//===----------------------------------------------------------------------===//

cv.and.sci.h t0, t1, 0
# CHECK-INSTR: cv.and.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sci.h t3, t4, -32
# CHECK-INSTR: cv.and.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sci.h a0, a1, 7
# CHECK-INSTR: cv.and.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x6a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sci.h s0, s1, -1
# CHECK-INSTR: cv.and.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x6b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.and.sci.b
//===----------------------------------------------------------------------===//

cv.and.sci.b t0, t1, 0
# CHECK-INSTR: cv.and.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x68]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sci.b t3, t4, -32
# CHECK-INSTR: cv.and.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x69]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sci.b a0, a1, 7
# CHECK-INSTR: cv.and.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x6a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.and.sci.b s0, s1, -1
# CHECK-INSTR: cv.and.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x6b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.abs.h
//===----------------------------------------------------------------------===//

cv.abs.h t0, t1
# CHECK-INSTR: cv.abs.h t0, t1
# CHECK-ENCODING: [0xfb,0x02,0x03,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.abs.h t3, t4
# CHECK-INSTR: cv.abs.h t3, t4
# CHECK-ENCODING: [0x7b,0x8e,0x0e,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.abs.h a0, a1
# CHECK-INSTR: cv.abs.h a0, a1
# CHECK-ENCODING: [0x7b,0x85,0x05,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.abs.h s0, s1
# CHECK-INSTR: cv.abs.h s0, s1
# CHECK-ENCODING: [0x7b,0x84,0x04,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.abs.b
//===----------------------------------------------------------------------===//

cv.abs.b t0, t1
# CHECK-INSTR: cv.abs.b t0, t1
# CHECK-ENCODING: [0xfb,0x12,0x03,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.abs.b t3, t4
# CHECK-INSTR: cv.abs.b t3, t4
# CHECK-ENCODING: [0x7b,0x9e,0x0e,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.abs.b a0, a1
# CHECK-INSTR: cv.abs.b a0, a1
# CHECK-ENCODING: [0x7b,0x95,0x05,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.abs.b s0, s1
# CHECK-INSTR: cv.abs.b s0, s1
# CHECK-ENCODING: [0x7b,0x94,0x04,0x70]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotup.h
//===----------------------------------------------------------------------===//

cv.dotup.h t0, t1, t2
# CHECK-INSTR: cv.dotup.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.h t3, t4, t5
# CHECK-INSTR: cv.dotup.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.h a0, a1, a2
# CHECK-INSTR: cv.dotup.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.h s0, s1, s2
# CHECK-INSTR: cv.dotup.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotup.b
//===----------------------------------------------------------------------===//

cv.dotup.b t0, t1, t2
# CHECK-INSTR: cv.dotup.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.b t3, t4, t5
# CHECK-INSTR: cv.dotup.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.b a0, a1, a2
# CHECK-INSTR: cv.dotup.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.b s0, s1, s2
# CHECK-INSTR: cv.dotup.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotup.sc.h
//===----------------------------------------------------------------------===//

cv.dotup.sc.h t0, t1, t2
# CHECK-INSTR: cv.dotup.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sc.h t3, t4, t5
# CHECK-INSTR: cv.dotup.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sc.h a0, a1, a2
# CHECK-INSTR: cv.dotup.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sc.h s0, s1, s2
# CHECK-INSTR: cv.dotup.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotup.sc.b
//===----------------------------------------------------------------------===//

cv.dotup.sc.b t0, t1, t2
# CHECK-INSTR: cv.dotup.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sc.b t3, t4, t5
# CHECK-INSTR: cv.dotup.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sc.b a0, a1, a2
# CHECK-INSTR: cv.dotup.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sc.b s0, s1, s2
# CHECK-INSTR: cv.dotup.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotup.sci.h
//===----------------------------------------------------------------------===//

cv.dotup.sci.h t0, t1, 0
# CHECK-INSTR: cv.dotup.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sci.h t3, t4, 32
# CHECK-INSTR: cv.dotup.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sci.h a0, a1, 7
# CHECK-INSTR: cv.dotup.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sci.h s0, s1, 63
# CHECK-INSTR: cv.dotup.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x83]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotup.sci.b
//===----------------------------------------------------------------------===//

cv.dotup.sci.b t0, t1, 0
# CHECK-INSTR: cv.dotup.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x80]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sci.b t3, t4, 32
# CHECK-INSTR: cv.dotup.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x81]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sci.b a0, a1, 7
# CHECK-INSTR: cv.dotup.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x82]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotup.sci.b s0, s1, 63
# CHECK-INSTR: cv.dotup.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x83]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotusp.h
//===----------------------------------------------------------------------===//

cv.dotusp.h t0, t1, t2
# CHECK-INSTR: cv.dotusp.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.h t3, t4, t5
# CHECK-INSTR: cv.dotusp.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.h a0, a1, a2
# CHECK-INSTR: cv.dotusp.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.h s0, s1, s2
# CHECK-INSTR: cv.dotusp.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotusp.b
//===----------------------------------------------------------------------===//

cv.dotusp.b t0, t1, t2
# CHECK-INSTR: cv.dotusp.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.b t3, t4, t5
# CHECK-INSTR: cv.dotusp.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.b a0, a1, a2
# CHECK-INSTR: cv.dotusp.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.b s0, s1, s2
# CHECK-INSTR: cv.dotusp.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotusp.sc.h
//===----------------------------------------------------------------------===//

cv.dotusp.sc.h t0, t1, t2
# CHECK-INSTR: cv.dotusp.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sc.h t3, t4, t5
# CHECK-INSTR: cv.dotusp.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sc.h a0, a1, a2
# CHECK-INSTR: cv.dotusp.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sc.h s0, s1, s2
# CHECK-INSTR: cv.dotusp.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotusp.sc.b
//===----------------------------------------------------------------------===//

cv.dotusp.sc.b t0, t1, t2
# CHECK-INSTR: cv.dotusp.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sc.b t3, t4, t5
# CHECK-INSTR: cv.dotusp.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sc.b a0, a1, a2
# CHECK-INSTR: cv.dotusp.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sc.b s0, s1, s2
# CHECK-INSTR: cv.dotusp.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotusp.sci.h
//===----------------------------------------------------------------------===//

cv.dotusp.sci.h t0, t1, 0
# CHECK-INSTR: cv.dotusp.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sci.h t3, t4, -32
# CHECK-INSTR: cv.dotusp.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sci.h a0, a1, 7
# CHECK-INSTR: cv.dotusp.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sci.h s0, s1, -1
# CHECK-INSTR: cv.dotusp.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x8b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotusp.sci.b
//===----------------------------------------------------------------------===//

cv.dotusp.sci.b t0, t1, 0
# CHECK-INSTR: cv.dotusp.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x88]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sci.b t3, t4, -32
# CHECK-INSTR: cv.dotusp.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x89]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sci.b a0, a1, 7
# CHECK-INSTR: cv.dotusp.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x8a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotusp.sci.b s0, s1, -1
# CHECK-INSTR: cv.dotusp.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x8b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotsp.h
//===----------------------------------------------------------------------===//

cv.dotsp.h t0, t1, t2
# CHECK-INSTR: cv.dotsp.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.h t3, t4, t5
# CHECK-INSTR: cv.dotsp.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.h a0, a1, a2
# CHECK-INSTR: cv.dotsp.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.h s0, s1, s2
# CHECK-INSTR: cv.dotsp.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotsp.b
//===----------------------------------------------------------------------===//

cv.dotsp.b t0, t1, t2
# CHECK-INSTR: cv.dotsp.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.b t3, t4, t5
# CHECK-INSTR: cv.dotsp.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.b a0, a1, a2
# CHECK-INSTR: cv.dotsp.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.b s0, s1, s2
# CHECK-INSTR: cv.dotsp.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotsp.sc.h
//===----------------------------------------------------------------------===//

cv.dotsp.sc.h t0, t1, t2
# CHECK-INSTR: cv.dotsp.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sc.h t3, t4, t5
# CHECK-INSTR: cv.dotsp.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sc.h a0, a1, a2
# CHECK-INSTR: cv.dotsp.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sc.h s0, s1, s2
# CHECK-INSTR: cv.dotsp.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotsp.sc.b
//===----------------------------------------------------------------------===//

cv.dotsp.sc.b t0, t1, t2
# CHECK-INSTR: cv.dotsp.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sc.b t3, t4, t5
# CHECK-INSTR: cv.dotsp.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sc.b a0, a1, a2
# CHECK-INSTR: cv.dotsp.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sc.b s0, s1, s2
# CHECK-INSTR: cv.dotsp.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotsp.sci.h
//===----------------------------------------------------------------------===//

cv.dotsp.sci.h t0, t1, 0
# CHECK-INSTR: cv.dotsp.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sci.h t3, t4, -32
# CHECK-INSTR: cv.dotsp.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sci.h a0, a1, 7
# CHECK-INSTR: cv.dotsp.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sci.h s0, s1, -1
# CHECK-INSTR: cv.dotsp.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x93]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.dotsp.sci.b
//===----------------------------------------------------------------------===//

cv.dotsp.sci.b t0, t1, 0
# CHECK-INSTR: cv.dotsp.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x90]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sci.b t3, t4, -32
# CHECK-INSTR: cv.dotsp.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x91]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sci.b a0, a1, 7
# CHECK-INSTR: cv.dotsp.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x92]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.dotsp.sci.b s0, s1, -1
# CHECK-INSTR: cv.dotsp.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x93]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotup.h
//===----------------------------------------------------------------------===//

cv.sdotup.h t0, t1, t2
# CHECK-INSTR: cv.sdotup.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.h t3, t4, t5
# CHECK-INSTR: cv.sdotup.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.h a0, a1, a2
# CHECK-INSTR: cv.sdotup.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.h s0, s1, s2
# CHECK-INSTR: cv.sdotup.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotup.b
//===----------------------------------------------------------------------===//

cv.sdotup.b t0, t1, t2
# CHECK-INSTR: cv.sdotup.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.b t3, t4, t5
# CHECK-INSTR: cv.sdotup.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.b a0, a1, a2
# CHECK-INSTR: cv.sdotup.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.b s0, s1, s2
# CHECK-INSTR: cv.sdotup.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotup.sc.h
//===----------------------------------------------------------------------===//

cv.sdotup.sc.h t0, t1, t2
# CHECK-INSTR: cv.sdotup.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sc.h t3, t4, t5
# CHECK-INSTR: cv.sdotup.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sc.h a0, a1, a2
# CHECK-INSTR: cv.sdotup.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sc.h s0, s1, s2
# CHECK-INSTR: cv.sdotup.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotup.sc.b
//===----------------------------------------------------------------------===//

cv.sdotup.sc.b t0, t1, t2
# CHECK-INSTR: cv.sdotup.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sc.b t3, t4, t5
# CHECK-INSTR: cv.sdotup.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sc.b a0, a1, a2
# CHECK-INSTR: cv.sdotup.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sc.b s0, s1, s2
# CHECK-INSTR: cv.sdotup.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotup.sci.h
//===----------------------------------------------------------------------===//

cv.sdotup.sci.h t0, t1, 0
# CHECK-INSTR: cv.sdotup.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sci.h t3, t4, 32
# CHECK-INSTR: cv.sdotup.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sci.h a0, a1, 7
# CHECK-INSTR: cv.sdotup.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sci.h s0, s1, 63
# CHECK-INSTR: cv.sdotup.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x9b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotup.sci.b
//===----------------------------------------------------------------------===//

cv.sdotup.sci.b t0, t1, 0
# CHECK-INSTR: cv.sdotup.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x98]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sci.b t3, t4, 32
# CHECK-INSTR: cv.sdotup.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x99]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sci.b a0, a1, 7
# CHECK-INSTR: cv.sdotup.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x9a]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotup.sci.b s0, s1, 63
# CHECK-INSTR: cv.sdotup.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x9b]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotusp.h
//===----------------------------------------------------------------------===//

cv.sdotusp.h t0, t1, t2
# CHECK-INSTR: cv.sdotusp.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.h t3, t4, t5
# CHECK-INSTR: cv.sdotusp.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.h a0, a1, a2
# CHECK-INSTR: cv.sdotusp.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.h s0, s1, s2
# CHECK-INSTR: cv.sdotusp.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotusp.b
//===----------------------------------------------------------------------===//

cv.sdotusp.b t0, t1, t2
# CHECK-INSTR: cv.sdotusp.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.b t3, t4, t5
# CHECK-INSTR: cv.sdotusp.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.b a0, a1, a2
# CHECK-INSTR: cv.sdotusp.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.b s0, s1, s2
# CHECK-INSTR: cv.sdotusp.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotusp.sc.h
//===----------------------------------------------------------------------===//

cv.sdotusp.sc.h t0, t1, t2
# CHECK-INSTR: cv.sdotusp.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sc.h t3, t4, t5
# CHECK-INSTR: cv.sdotusp.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sc.h a0, a1, a2
# CHECK-INSTR: cv.sdotusp.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sc.h s0, s1, s2
# CHECK-INSTR: cv.sdotusp.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotusp.sc.b
//===----------------------------------------------------------------------===//

cv.sdotusp.sc.b t0, t1, t2
# CHECK-INSTR: cv.sdotusp.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sc.b t3, t4, t5
# CHECK-INSTR: cv.sdotusp.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sc.b a0, a1, a2
# CHECK-INSTR: cv.sdotusp.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sc.b s0, s1, s2
# CHECK-INSTR: cv.sdotusp.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotusp.sci.h
//===----------------------------------------------------------------------===//

cv.sdotusp.sci.h t0, t1, 0
# CHECK-INSTR: cv.sdotusp.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sci.h t3, t4, -32
# CHECK-INSTR: cv.sdotusp.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sci.h a0, a1, 7
# CHECK-INSTR: cv.sdotusp.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sci.h s0, s1, -1
# CHECK-INSTR: cv.sdotusp.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0xa3]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotusp.sci.b
//===----------------------------------------------------------------------===//

cv.sdotusp.sci.b t0, t1, 0
# CHECK-INSTR: cv.sdotusp.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0xa0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sci.b t3, t4, -32
# CHECK-INSTR: cv.sdotusp.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0xa1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sci.b a0, a1, 7
# CHECK-INSTR: cv.sdotusp.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0xa2]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotusp.sci.b s0, s1, -1
# CHECK-INSTR: cv.sdotusp.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0xa3]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotsp.h
//===----------------------------------------------------------------------===//

cv.sdotsp.h t0, t1, t2
# CHECK-INSTR: cv.sdotsp.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.h t3, t4, t5
# CHECK-INSTR: cv.sdotsp.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.h a0, a1, a2
# CHECK-INSTR: cv.sdotsp.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.h s0, s1, s2
# CHECK-INSTR: cv.sdotsp.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotsp.b
//===----------------------------------------------------------------------===//

cv.sdotsp.b t0, t1, t2
# CHECK-INSTR: cv.sdotsp.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.b t3, t4, t5
# CHECK-INSTR: cv.sdotsp.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.b a0, a1, a2
# CHECK-INSTR: cv.sdotsp.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.b s0, s1, s2
# CHECK-INSTR: cv.sdotsp.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotsp.sc.h
//===----------------------------------------------------------------------===//

cv.sdotsp.sc.h t0, t1, t2
# CHECK-INSTR: cv.sdotsp.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sc.h t3, t4, t5
# CHECK-INSTR: cv.sdotsp.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sc.h a0, a1, a2
# CHECK-INSTR: cv.sdotsp.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sc.h s0, s1, s2
# CHECK-INSTR: cv.sdotsp.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotsp.sc.b
//===----------------------------------------------------------------------===//

cv.sdotsp.sc.b t0, t1, t2
# CHECK-INSTR: cv.sdotsp.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sc.b t3, t4, t5
# CHECK-INSTR: cv.sdotsp.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sc.b a0, a1, a2
# CHECK-INSTR: cv.sdotsp.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sc.b s0, s1, s2
# CHECK-INSTR: cv.sdotsp.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotsp.sci.h
//===----------------------------------------------------------------------===//

cv.sdotsp.sci.h t0, t1, 0
# CHECK-INSTR: cv.sdotsp.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sci.h t3, t4, -32
# CHECK-INSTR: cv.sdotsp.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sci.h a0, a1, 7
# CHECK-INSTR: cv.sdotsp.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sci.h s0, s1, -1
# CHECK-INSTR: cv.sdotsp.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0xab]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sdotsp.sci.b
//===----------------------------------------------------------------------===//

cv.sdotsp.sci.b t0, t1, 0
# CHECK-INSTR: cv.sdotsp.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0xa8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sci.b t3, t4, -32
# CHECK-INSTR: cv.sdotsp.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0xa9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sci.b a0, a1, 7
# CHECK-INSTR: cv.sdotsp.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0xaa]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sdotsp.sci.b s0, s1, -1
# CHECK-INSTR: cv.sdotsp.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0xab]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.extract.h
//===----------------------------------------------------------------------===//

cv.extract.h t0, t1, 0
# CHECK-INSTR: cv.extract.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x02,0x03,0xb8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extract.h t3, t4, 32
# CHECK-INSTR: cv.extract.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0x8e,0x0e,0xb9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extract.h a0, a1, 7
# CHECK-INSTR: cv.extract.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0x85,0x35,0xba]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extract.h s0, s1, 63
# CHECK-INSTR: cv.extract.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0x84,0xf4,0xbb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.extract.b
//===----------------------------------------------------------------------===//

cv.extract.b t0, t1, 0
# CHECK-INSTR: cv.extract.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x12,0x03,0xb8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extract.b t3, t4, 32
# CHECK-INSTR: cv.extract.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0x9e,0x0e,0xb9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extract.b a0, a1, 7
# CHECK-INSTR: cv.extract.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0x95,0x35,0xba]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extract.b s0, s1, 63
# CHECK-INSTR: cv.extract.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0x94,0xf4,0xbb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.extractu.h
//===----------------------------------------------------------------------===//

cv.extractu.h t0, t1, 0
# CHECK-INSTR: cv.extractu.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x22,0x03,0xb8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extractu.h t3, t4, 32
# CHECK-INSTR: cv.extractu.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xae,0x0e,0xb9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extractu.h a0, a1, 7
# CHECK-INSTR: cv.extractu.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xa5,0x35,0xba]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extractu.h s0, s1, 63
# CHECK-INSTR: cv.extractu.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xa4,0xf4,0xbb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.extractu.b
//===----------------------------------------------------------------------===//

cv.extractu.b t0, t1, 0
# CHECK-INSTR: cv.extractu.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x32,0x03,0xb8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extractu.b t3, t4, 32
# CHECK-INSTR: cv.extractu.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xbe,0x0e,0xb9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extractu.b a0, a1, 7
# CHECK-INSTR: cv.extractu.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xb5,0x35,0xba]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.extractu.b s0, s1, 63
# CHECK-INSTR: cv.extractu.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xb4,0xf4,0xbb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.insert.h
//===----------------------------------------------------------------------===//

cv.insert.h t0, t1, 0
# CHECK-INSTR: cv.insert.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x42,0x03,0xb8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.insert.h t3, t4, 32
# CHECK-INSTR: cv.insert.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xce,0x0e,0xb9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.insert.h a0, a1, 7
# CHECK-INSTR: cv.insert.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xc5,0x35,0xba]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.insert.h s0, s1, 63
# CHECK-INSTR: cv.insert.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xc4,0xf4,0xbb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.insert.b
//===----------------------------------------------------------------------===//

cv.insert.b t0, t1, 0
# CHECK-INSTR: cv.insert.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x52,0x03,0xb8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.insert.b t3, t4, 32
# CHECK-INSTR: cv.insert.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xde,0x0e,0xb9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.insert.b a0, a1, 7
# CHECK-INSTR: cv.insert.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xd5,0x35,0xba]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.insert.b s0, s1, 63
# CHECK-INSTR: cv.insert.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xd4,0xf4,0xbb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffle.h
//===----------------------------------------------------------------------===//

cv.shuffle.h t0, t1, t2
# CHECK-INSTR: cv.shuffle.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.h t3, t4, t5
# CHECK-INSTR: cv.shuffle.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0xc1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.h a0, a1, a2
# CHECK-INSTR: cv.shuffle.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.h s0, s1, s2
# CHECK-INSTR: cv.shuffle.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0xc1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffle.b
//===----------------------------------------------------------------------===//

cv.shuffle.b t0, t1, t2
# CHECK-INSTR: cv.shuffle.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.b t3, t4, t5
# CHECK-INSTR: cv.shuffle.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0xc1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.b a0, a1, a2
# CHECK-INSTR: cv.shuffle.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.b s0, s1, s2
# CHECK-INSTR: cv.shuffle.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0xc1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffle.sci.h
//===----------------------------------------------------------------------===//

cv.shuffle.sci.h t0, t1, 0
# CHECK-INSTR: cv.shuffle.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.sci.h t3, t4, 32
# CHECK-INSTR: cv.shuffle.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0xc1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.sci.h a0, a1, 7
# CHECK-INSTR: cv.shuffle.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0xc2]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle.sci.h s0, s1, 63
# CHECK-INSTR: cv.shuffle.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0xc3]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffleI0.sci.b
//===----------------------------------------------------------------------===//

cv.shufflei0.sci.b t0, t1, 0
# CHECK-INSTR: cv.shufflei0.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0xc0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei0.sci.b t3, t4, 32
# CHECK-INSTR: cv.shufflei0.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0xc1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei0.sci.b a0, a1, 7
# CHECK-INSTR: cv.shufflei0.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0xc2]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei0.sci.b s0, s1, 63
# CHECK-INSTR: cv.shufflei0.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0xc3]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffleI1.sci.b
//===----------------------------------------------------------------------===//

cv.shufflei1.sci.b t0, t1, 0
# CHECK-INSTR: cv.shufflei1.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0xc8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei1.sci.b t3, t4, 32
# CHECK-INSTR: cv.shufflei1.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0xc9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei1.sci.b a0, a1, 7
# CHECK-INSTR: cv.shufflei1.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0xca]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei1.sci.b s0, s1, 63
# CHECK-INSTR: cv.shufflei1.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0xcb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffleI2.sci.b
//===----------------------------------------------------------------------===//

cv.shufflei2.sci.b t0, t1, 0
# CHECK-INSTR: cv.shufflei2.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0xd0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei2.sci.b t3, t4, 32
# CHECK-INSTR: cv.shufflei2.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0xd1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei2.sci.b a0, a1, 7
# CHECK-INSTR: cv.shufflei2.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0xd2]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei2.sci.b s0, s1, 63
# CHECK-INSTR: cv.shufflei2.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0xd3]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffleI3.sci.b
//===----------------------------------------------------------------------===//

cv.shufflei3.sci.b t0, t1, 0
# CHECK-INSTR: cv.shufflei3.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0xd8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei3.sci.b t3, t4, 32
# CHECK-INSTR: cv.shufflei3.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0xd9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei3.sci.b a0, a1, 7
# CHECK-INSTR: cv.shufflei3.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0xda]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shufflei3.sci.b s0, s1, 63
# CHECK-INSTR: cv.shufflei3.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0xdb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffle2.h
//===----------------------------------------------------------------------===//

cv.shuffle2.h t0, t1, t2
# CHECK-INSTR: cv.shuffle2.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle2.h t3, t4, t5
# CHECK-INSTR: cv.shuffle2.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0xe1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle2.h a0, a1, a2
# CHECK-INSTR: cv.shuffle2.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle2.h s0, s1, s2
# CHECK-INSTR: cv.shuffle2.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0xe1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.shuffle2.b
//===----------------------------------------------------------------------===//

cv.shuffle2.b t0, t1, t2
# CHECK-INSTR: cv.shuffle2.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle2.b t3, t4, t5
# CHECK-INSTR: cv.shuffle2.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0xe1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle2.b a0, a1, a2
# CHECK-INSTR: cv.shuffle2.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0xe0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.shuffle2.b s0, s1, s2
# CHECK-INSTR: cv.shuffle2.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0xe1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.pack
//===----------------------------------------------------------------------===//

cv.pack t0, t1, t2
# CHECK-INSTR: cv.pack t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0xf0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.pack t3, t4, t5
# CHECK-INSTR: cv.pack t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0xf1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.pack a0, a1, a2
# CHECK-INSTR: cv.pack a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0xf0]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.pack s0, s1, s2
# CHECK-INSTR: cv.pack s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0xf1]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.pack.h
//===----------------------------------------------------------------------===//

cv.pack.h t0, t1, t2
# CHECK-INSTR: cv.pack.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0xf2]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.pack.h t3, t4, t5
# CHECK-INSTR: cv.pack.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0xf3]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.pack.h a0, a1, a2
# CHECK-INSTR: cv.pack.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0xf2]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.pack.h s0, s1, s2
# CHECK-INSTR: cv.pack.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0xf3]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.packhi.b
//===----------------------------------------------------------------------===//

cv.packhi.b t0, t1, t2
# CHECK-INSTR: cv.packhi.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0xfa]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.packhi.b t3, t4, t5
# CHECK-INSTR: cv.packhi.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0xfb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.packhi.b a0, a1, a2
# CHECK-INSTR: cv.packhi.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0xfa]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.packhi.b s0, s1, s2
# CHECK-INSTR: cv.packhi.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0xfb]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.packlo.b
//===----------------------------------------------------------------------===//

cv.packlo.b t0, t1, t2
# CHECK-INSTR: cv.packlo.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0xf8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.packlo.b t3, t4, t5
# CHECK-INSTR: cv.packlo.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0xf9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.packlo.b a0, a1, a2
# CHECK-INSTR: cv.packlo.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0xf8]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.packlo.b s0, s1, s2
# CHECK-INSTR: cv.packlo.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0xf9]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpeq.h
//===----------------------------------------------------------------------===//

cv.cmpeq.h t0, t1, t2
# CHECK-INSTR: cv.cmpeq.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.h t3, t4, t5
# CHECK-INSTR: cv.cmpeq.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.h a0, a1, a2
# CHECK-INSTR: cv.cmpeq.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.h s0, s1, s2
# CHECK-INSTR: cv.cmpeq.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpeq.b
//===----------------------------------------------------------------------===//

cv.cmpeq.b t0, t1, t2
# CHECK-INSTR: cv.cmpeq.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.b t3, t4, t5
# CHECK-INSTR: cv.cmpeq.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.b a0, a1, a2
# CHECK-INSTR: cv.cmpeq.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.b s0, s1, s2
# CHECK-INSTR: cv.cmpeq.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpeq.sc.h
//===----------------------------------------------------------------------===//

cv.cmpeq.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmpeq.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmpeq.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmpeq.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmpeq.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpeq.sc.b
//===----------------------------------------------------------------------===//

cv.cmpeq.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmpeq.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmpeq.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmpeq.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmpeq.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpeq.sci.h
//===----------------------------------------------------------------------===//

cv.cmpeq.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmpeq.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sci.h t3, t4, -32
# CHECK-INSTR: cv.cmpeq.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmpeq.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x06]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sci.h s0, s1, -1
# CHECK-INSTR: cv.cmpeq.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x07]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpeq.sci.b
//===----------------------------------------------------------------------===//

cv.cmpeq.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmpeq.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x04]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sci.b t3, t4, -32
# CHECK-INSTR: cv.cmpeq.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x05]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmpeq.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x06]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpeq.sci.b s0, s1, -1
# CHECK-INSTR: cv.cmpeq.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x07]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpne.h
//===----------------------------------------------------------------------===//

cv.cmpne.h t0, t1, t2
# CHECK-INSTR: cv.cmpne.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.h t3, t4, t5
# CHECK-INSTR: cv.cmpne.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.h a0, a1, a2
# CHECK-INSTR: cv.cmpne.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.h s0, s1, s2
# CHECK-INSTR: cv.cmpne.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpne.b
//===----------------------------------------------------------------------===//

cv.cmpne.b t0, t1, t2
# CHECK-INSTR: cv.cmpne.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.b t3, t4, t5
# CHECK-INSTR: cv.cmpne.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.b a0, a1, a2
# CHECK-INSTR: cv.cmpne.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.b s0, s1, s2
# CHECK-INSTR: cv.cmpne.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpne.sc.h
//===----------------------------------------------------------------------===//

cv.cmpne.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmpne.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmpne.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmpne.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmpne.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpne.sc.b
//===----------------------------------------------------------------------===//

cv.cmpne.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmpne.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmpne.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmpne.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmpne.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpne.sci.h
//===----------------------------------------------------------------------===//

cv.cmpne.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmpne.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sci.h t3, t4, -32
# CHECK-INSTR: cv.cmpne.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmpne.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x0e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sci.h s0, s1, -1
# CHECK-INSTR: cv.cmpne.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x0f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpne.sci.b
//===----------------------------------------------------------------------===//

cv.cmpne.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmpne.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x0c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sci.b t3, t4, -32
# CHECK-INSTR: cv.cmpne.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x0d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmpne.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x0e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpne.sci.b s0, s1, -1
# CHECK-INSTR: cv.cmpne.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x0f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgt.h
//===----------------------------------------------------------------------===//

cv.cmpgt.h t0, t1, t2
# CHECK-INSTR: cv.cmpgt.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.h t3, t4, t5
# CHECK-INSTR: cv.cmpgt.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.h a0, a1, a2
# CHECK-INSTR: cv.cmpgt.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.h s0, s1, s2
# CHECK-INSTR: cv.cmpgt.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgt.b
//===----------------------------------------------------------------------===//

cv.cmpgt.b t0, t1, t2
# CHECK-INSTR: cv.cmpgt.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.b t3, t4, t5
# CHECK-INSTR: cv.cmpgt.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.b a0, a1, a2
# CHECK-INSTR: cv.cmpgt.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.b s0, s1, s2
# CHECK-INSTR: cv.cmpgt.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgt.sc.h
//===----------------------------------------------------------------------===//

cv.cmpgt.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmpgt.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmpgt.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmpgt.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmpgt.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgt.sc.b
//===----------------------------------------------------------------------===//

cv.cmpgt.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmpgt.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmpgt.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmpgt.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmpgt.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgt.sci.h
//===----------------------------------------------------------------------===//

cv.cmpgt.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmpgt.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sci.h t3, t4, -32
# CHECK-INSTR: cv.cmpgt.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmpgt.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x16]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sci.h s0, s1, -1
# CHECK-INSTR: cv.cmpgt.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x17]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgt.sci.b
//===----------------------------------------------------------------------===//

cv.cmpgt.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmpgt.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x14]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sci.b t3, t4, -32
# CHECK-INSTR: cv.cmpgt.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x15]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmpgt.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x16]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgt.sci.b s0, s1, -1
# CHECK-INSTR: cv.cmpgt.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x17]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpge.h
//===----------------------------------------------------------------------===//

cv.cmpge.h t0, t1, t2
# CHECK-INSTR: cv.cmpge.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.h t3, t4, t5
# CHECK-INSTR: cv.cmpge.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.h a0, a1, a2
# CHECK-INSTR: cv.cmpge.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.h s0, s1, s2
# CHECK-INSTR: cv.cmpge.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpge.b
//===----------------------------------------------------------------------===//

cv.cmpge.b t0, t1, t2
# CHECK-INSTR: cv.cmpge.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.b t3, t4, t5
# CHECK-INSTR: cv.cmpge.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.b a0, a1, a2
# CHECK-INSTR: cv.cmpge.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.b s0, s1, s2
# CHECK-INSTR: cv.cmpge.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpge.sc.h
//===----------------------------------------------------------------------===//

cv.cmpge.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmpge.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmpge.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmpge.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmpge.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpge.sc.b
//===----------------------------------------------------------------------===//

cv.cmpge.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmpge.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmpge.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmpge.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmpge.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpge.sci.h
//===----------------------------------------------------------------------===//

cv.cmpge.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmpge.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sci.h t3, t4, -32
# CHECK-INSTR: cv.cmpge.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmpge.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x1e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sci.h s0, s1, -1
# CHECK-INSTR: cv.cmpge.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x1f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpge.sci.b
//===----------------------------------------------------------------------===//

cv.cmpge.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmpge.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x1c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sci.b t3, t4, -32
# CHECK-INSTR: cv.cmpge.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x1d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmpge.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x1e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpge.sci.b s0, s1, -1
# CHECK-INSTR: cv.cmpge.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x1f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmplt.h
//===----------------------------------------------------------------------===//

cv.cmplt.h t0, t1, t2
# CHECK-INSTR: cv.cmplt.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.h t3, t4, t5
# CHECK-INSTR: cv.cmplt.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.h a0, a1, a2
# CHECK-INSTR: cv.cmplt.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.h s0, s1, s2
# CHECK-INSTR: cv.cmplt.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmplt.b
//===----------------------------------------------------------------------===//

cv.cmplt.b t0, t1, t2
# CHECK-INSTR: cv.cmplt.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.b t3, t4, t5
# CHECK-INSTR: cv.cmplt.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.b a0, a1, a2
# CHECK-INSTR: cv.cmplt.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.b s0, s1, s2
# CHECK-INSTR: cv.cmplt.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmplt.sc.h
//===----------------------------------------------------------------------===//

cv.cmplt.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmplt.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmplt.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmplt.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmplt.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmplt.sc.b
//===----------------------------------------------------------------------===//

cv.cmplt.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmplt.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmplt.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmplt.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmplt.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmplt.sci.h
//===----------------------------------------------------------------------===//

cv.cmplt.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmplt.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sci.h t3, t4, -32
# CHECK-INSTR: cv.cmplt.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmplt.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x26]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sci.h s0, s1, -1
# CHECK-INSTR: cv.cmplt.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x27]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmplt.sci.b
//===----------------------------------------------------------------------===//

cv.cmplt.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmplt.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x24]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sci.b t3, t4, -32
# CHECK-INSTR: cv.cmplt.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x25]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmplt.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x26]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmplt.sci.b s0, s1, -1
# CHECK-INSTR: cv.cmplt.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x27]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmple.h
//===----------------------------------------------------------------------===//

cv.cmple.h t0, t1, t2
# CHECK-INSTR: cv.cmple.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.h t3, t4, t5
# CHECK-INSTR: cv.cmple.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.h a0, a1, a2
# CHECK-INSTR: cv.cmple.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.h s0, s1, s2
# CHECK-INSTR: cv.cmple.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmple.b
//===----------------------------------------------------------------------===//

cv.cmple.b t0, t1, t2
# CHECK-INSTR: cv.cmple.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.b t3, t4, t5
# CHECK-INSTR: cv.cmple.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.b a0, a1, a2
# CHECK-INSTR: cv.cmple.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.b s0, s1, s2
# CHECK-INSTR: cv.cmple.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmple.sc.h
//===----------------------------------------------------------------------===//

cv.cmple.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmple.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmple.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmple.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmple.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmple.sc.b
//===----------------------------------------------------------------------===//

cv.cmple.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmple.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmple.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmple.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmple.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmple.sci.h
//===----------------------------------------------------------------------===//

cv.cmple.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmple.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sci.h t3, t4, -32
# CHECK-INSTR: cv.cmple.sci.h t3, t4, -32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmple.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x2e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sci.h s0, s1, -1
# CHECK-INSTR: cv.cmple.sci.h s0, s1, -1
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x2f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmple.sci.b
//===----------------------------------------------------------------------===//

cv.cmple.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmple.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x2c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sci.b t3, t4, -32
# CHECK-INSTR: cv.cmple.sci.b t3, t4, -32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x2d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmple.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x2e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmple.sci.b s0, s1, -1
# CHECK-INSTR: cv.cmple.sci.b s0, s1, -1
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x2f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgtu.h
//===----------------------------------------------------------------------===//

cv.cmpgtu.h t0, t1, t2
# CHECK-INSTR: cv.cmpgtu.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.h t3, t4, t5
# CHECK-INSTR: cv.cmpgtu.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.h a0, a1, a2
# CHECK-INSTR: cv.cmpgtu.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.h s0, s1, s2
# CHECK-INSTR: cv.cmpgtu.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgtu.b
//===----------------------------------------------------------------------===//

cv.cmpgtu.b t0, t1, t2
# CHECK-INSTR: cv.cmpgtu.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.b t3, t4, t5
# CHECK-INSTR: cv.cmpgtu.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.b a0, a1, a2
# CHECK-INSTR: cv.cmpgtu.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.b s0, s1, s2
# CHECK-INSTR: cv.cmpgtu.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgtu.sc.h
//===----------------------------------------------------------------------===//

cv.cmpgtu.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmpgtu.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmpgtu.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmpgtu.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmpgtu.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgtu.sc.b
//===----------------------------------------------------------------------===//

cv.cmpgtu.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmpgtu.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmpgtu.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmpgtu.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmpgtu.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgtu.sci.h
//===----------------------------------------------------------------------===//

cv.cmpgtu.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmpgtu.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sci.h t3, t4, 32
# CHECK-INSTR: cv.cmpgtu.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmpgtu.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x36]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sci.h s0, s1, 63
# CHECK-INSTR: cv.cmpgtu.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x37]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgtu.sci.b
//===----------------------------------------------------------------------===//

cv.cmpgtu.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmpgtu.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x34]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sci.b t3, t4, 32
# CHECK-INSTR: cv.cmpgtu.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x35]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmpgtu.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x36]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgtu.sci.b s0, s1, 63
# CHECK-INSTR: cv.cmpgtu.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x37]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgeu.h
//===----------------------------------------------------------------------===//

cv.cmpgeu.h t0, t1, t2
# CHECK-INSTR: cv.cmpgeu.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.h t3, t4, t5
# CHECK-INSTR: cv.cmpgeu.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.h a0, a1, a2
# CHECK-INSTR: cv.cmpgeu.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.h s0, s1, s2
# CHECK-INSTR: cv.cmpgeu.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgeu.b
//===----------------------------------------------------------------------===//

cv.cmpgeu.b t0, t1, t2
# CHECK-INSTR: cv.cmpgeu.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.b t3, t4, t5
# CHECK-INSTR: cv.cmpgeu.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.b a0, a1, a2
# CHECK-INSTR: cv.cmpgeu.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.b s0, s1, s2
# CHECK-INSTR: cv.cmpgeu.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgeu.sc.h
//===----------------------------------------------------------------------===//

cv.cmpgeu.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmpgeu.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmpgeu.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmpgeu.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmpgeu.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgeu.sc.b
//===----------------------------------------------------------------------===//

cv.cmpgeu.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmpgeu.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmpgeu.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmpgeu.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmpgeu.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgeu.sci.h
//===----------------------------------------------------------------------===//

cv.cmpgeu.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmpgeu.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sci.h t3, t4, 32
# CHECK-INSTR: cv.cmpgeu.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmpgeu.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x3e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sci.h s0, s1, 63
# CHECK-INSTR: cv.cmpgeu.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x3f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpgeu.sci.b
//===----------------------------------------------------------------------===//

cv.cmpgeu.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmpgeu.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x3c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sci.b t3, t4, 32
# CHECK-INSTR: cv.cmpgeu.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x3d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmpgeu.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x3e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpgeu.sci.b s0, s1, 63
# CHECK-INSTR: cv.cmpgeu.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x3f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpltu.h
//===----------------------------------------------------------------------===//

cv.cmpltu.h t0, t1, t2
# CHECK-INSTR: cv.cmpltu.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.h t3, t4, t5
# CHECK-INSTR: cv.cmpltu.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.h a0, a1, a2
# CHECK-INSTR: cv.cmpltu.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.h s0, s1, s2
# CHECK-INSTR: cv.cmpltu.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpltu.b
//===----------------------------------------------------------------------===//

cv.cmpltu.b t0, t1, t2
# CHECK-INSTR: cv.cmpltu.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.b t3, t4, t5
# CHECK-INSTR: cv.cmpltu.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.b a0, a1, a2
# CHECK-INSTR: cv.cmpltu.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.b s0, s1, s2
# CHECK-INSTR: cv.cmpltu.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpltu.sc.h
//===----------------------------------------------------------------------===//

cv.cmpltu.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmpltu.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmpltu.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmpltu.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmpltu.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpltu.sc.b
//===----------------------------------------------------------------------===//

cv.cmpltu.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmpltu.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmpltu.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmpltu.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmpltu.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpltu.sci.h
//===----------------------------------------------------------------------===//

cv.cmpltu.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmpltu.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sci.h t3, t4, 32
# CHECK-INSTR: cv.cmpltu.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmpltu.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x46]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sci.h s0, s1, 63
# CHECK-INSTR: cv.cmpltu.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x47]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpltu.sci.b
//===----------------------------------------------------------------------===//

cv.cmpltu.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmpltu.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x44]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sci.b t3, t4, 32
# CHECK-INSTR: cv.cmpltu.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x45]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmpltu.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x46]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpltu.sci.b s0, s1, 63
# CHECK-INSTR: cv.cmpltu.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x47]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpleu.h
//===----------------------------------------------------------------------===//

cv.cmpleu.h t0, t1, t2
# CHECK-INSTR: cv.cmpleu.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.h t3, t4, t5
# CHECK-INSTR: cv.cmpleu.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.h a0, a1, a2
# CHECK-INSTR: cv.cmpleu.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.h s0, s1, s2
# CHECK-INSTR: cv.cmpleu.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpleu.b
//===----------------------------------------------------------------------===//

cv.cmpleu.b t0, t1, t2
# CHECK-INSTR: cv.cmpleu.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x12,0x73,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.b t3, t4, t5
# CHECK-INSTR: cv.cmpleu.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0x9e,0xee,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.b a0, a1, a2
# CHECK-INSTR: cv.cmpleu.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0x95,0xc5,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.b s0, s1, s2
# CHECK-INSTR: cv.cmpleu.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0x94,0x24,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpleu.sc.h
//===----------------------------------------------------------------------===//

cv.cmpleu.sc.h t0, t1, t2
# CHECK-INSTR: cv.cmpleu.sc.h t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sc.h t3, t4, t5
# CHECK-INSTR: cv.cmpleu.sc.h t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sc.h a0, a1, a2
# CHECK-INSTR: cv.cmpleu.sc.h a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sc.h s0, s1, s2
# CHECK-INSTR: cv.cmpleu.sc.h s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpleu.sc.b
//===----------------------------------------------------------------------===//

cv.cmpleu.sc.b t0, t1, t2
# CHECK-INSTR: cv.cmpleu.sc.b t0, t1, t2
# CHECK-ENCODING: [0xfb,0x52,0x73,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sc.b t3, t4, t5
# CHECK-INSTR: cv.cmpleu.sc.b t3, t4, t5
# CHECK-ENCODING: [0x7b,0xde,0xee,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sc.b a0, a1, a2
# CHECK-INSTR: cv.cmpleu.sc.b a0, a1, a2
# CHECK-ENCODING: [0x7b,0xd5,0xc5,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sc.b s0, s1, s2
# CHECK-INSTR: cv.cmpleu.sc.b s0, s1, s2
# CHECK-ENCODING: [0x7b,0xd4,0x24,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpleu.sci.h
//===----------------------------------------------------------------------===//

cv.cmpleu.sci.h t0, t1, 0
# CHECK-INSTR: cv.cmpleu.sci.h t0, t1, 0
# CHECK-ENCODING: [0xfb,0x62,0x03,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sci.h t3, t4, 32
# CHECK-INSTR: cv.cmpleu.sci.h t3, t4, 32
# CHECK-ENCODING: [0x7b,0xee,0x0e,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sci.h a0, a1, 7
# CHECK-INSTR: cv.cmpleu.sci.h a0, a1, 7
# CHECK-ENCODING: [0x7b,0xe5,0x35,0x4e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sci.h s0, s1, 63
# CHECK-INSTR: cv.cmpleu.sci.h s0, s1, 63
# CHECK-ENCODING: [0x7b,0xe4,0xf4,0x4f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cmpleu.sci.b
//===----------------------------------------------------------------------===//

cv.cmpleu.sci.b t0, t1, 0
# CHECK-INSTR: cv.cmpleu.sci.b t0, t1, 0
# CHECK-ENCODING: [0xfb,0x72,0x03,0x4c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sci.b t3, t4, 32
# CHECK-INSTR: cv.cmpleu.sci.b t3, t4, 32
# CHECK-ENCODING: [0x7b,0xfe,0x0e,0x4d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sci.b a0, a1, 7
# CHECK-INSTR: cv.cmpleu.sci.b a0, a1, 7
# CHECK-ENCODING: [0x7b,0xf5,0x35,0x4e]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cmpleu.sci.b s0, s1, 63
# CHECK-INSTR: cv.cmpleu.sci.b s0, s1, 63
# CHECK-ENCODING: [0x7b,0xf4,0xf4,0x4f]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxmul.r
//===----------------------------------------------------------------------===//

cv.cplxmul.r t0, t1, t2
# CHECK-INSTR: cv.cplxmul.r t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r t3, t4, t5
# CHECK-INSTR: cv.cplxmul.r t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x55]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r a0, a1, a2
# CHECK-INSTR: cv.cplxmul.r a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r s0, s1, s2
# CHECK-INSTR: cv.cplxmul.r s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x55]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxmul.i
//===----------------------------------------------------------------------===//

cv.cplxmul.i t0, t1, t2
# CHECK-INSTR: cv.cplxmul.i t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i t3, t4, t5
# CHECK-INSTR: cv.cplxmul.i t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x57]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i a0, a1, a2
# CHECK-INSTR: cv.cplxmul.i a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i s0, s1, s2
# CHECK-INSTR: cv.cplxmul.i s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x57]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxmul.r.div2
//===----------------------------------------------------------------------===//

cv.cplxmul.r.div2 t0, t1, t2
# CHECK-INSTR: cv.cplxmul.r.div2 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x22,0x73,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div2 t3, t4, t5
# CHECK-INSTR: cv.cplxmul.r.div2 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xae,0xee,0x55]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div2 a0, a1, a2
# CHECK-INSTR: cv.cplxmul.r.div2 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xa5,0xc5,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div2 s0, s1, s2
# CHECK-INSTR: cv.cplxmul.r.div2 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xa4,0x24,0x55]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxmul.i.div2
//===----------------------------------------------------------------------===//

cv.cplxmul.i.div2 t0, t1, t2
# CHECK-INSTR: cv.cplxmul.i.div2 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x22,0x73,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div2 t3, t4, t5
# CHECK-INSTR: cv.cplxmul.i.div2 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xae,0xee,0x57]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div2 a0, a1, a2
# CHECK-INSTR: cv.cplxmul.i.div2 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xa5,0xc5,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div2 s0, s1, s2
# CHECK-INSTR: cv.cplxmul.i.div2 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xa4,0x24,0x57]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxmul.r.div4
//===----------------------------------------------------------------------===//

cv.cplxmul.r.div4 t0, t1, t2
# CHECK-INSTR: cv.cplxmul.r.div4 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div4 t3, t4, t5
# CHECK-INSTR: cv.cplxmul.r.div4 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x55]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div4 a0, a1, a2
# CHECK-INSTR: cv.cplxmul.r.div4 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div4 s0, s1, s2
# CHECK-INSTR: cv.cplxmul.r.div4 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x55]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxmul.i.div4
//===----------------------------------------------------------------------===//

cv.cplxmul.i.div4 t0, t1, t2
# CHECK-INSTR: cv.cplxmul.i.div4 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div4 t3, t4, t5
# CHECK-INSTR: cv.cplxmul.i.div4 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x57]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div4 a0, a1, a2
# CHECK-INSTR: cv.cplxmul.i.div4 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div4 s0, s1, s2
# CHECK-INSTR: cv.cplxmul.i.div4 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x57]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxmul.r.div8
//===----------------------------------------------------------------------===//

cv.cplxmul.r.div8 t0, t1, t2
# CHECK-INSTR: cv.cplxmul.r.div8 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x62,0x73,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div8 t3, t4, t5
# CHECK-INSTR: cv.cplxmul.r.div8 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xee,0xee,0x55]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div8 a0, a1, a2
# CHECK-INSTR: cv.cplxmul.r.div8 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xe5,0xc5,0x54]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.r.div8 s0, s1, s2
# CHECK-INSTR: cv.cplxmul.r.div8 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xe4,0x24,0x55]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxmul.i.div8
//===----------------------------------------------------------------------===//

cv.cplxmul.i.div8 t0, t1, t2
# CHECK-INSTR: cv.cplxmul.i.div8 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x62,0x73,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div8 t3, t4, t5
# CHECK-INSTR: cv.cplxmul.i.div8 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xee,0xee,0x57]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div8 a0, a1, a2
# CHECK-INSTR: cv.cplxmul.i.div8 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xe5,0xc5,0x56]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxmul.i.div8 s0, s1, s2
# CHECK-INSTR: cv.cplxmul.i.div8 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xe4,0x24,0x57]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.cplxconj
//===----------------------------------------------------------------------===//

cv.cplxconj t0, t1
# CHECK-INSTR: cv.cplxconj t0, t1
# CHECK-ENCODING: [0xfb,0x02,0x03,0x5c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxconj t3, t4
# CHECK-INSTR: cv.cplxconj t3, t4
# CHECK-ENCODING: [0x7b,0x8e,0x0e,0x5c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxconj a0, a1
# CHECK-INSTR: cv.cplxconj a0, a1
# CHECK-ENCODING: [0x7b,0x85,0x05,0x5c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.cplxconj s0, s1
# CHECK-INSTR: cv.cplxconj s0, s1
# CHECK-ENCODING: [0x7b,0x84,0x04,0x5c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.subrotmj
//===----------------------------------------------------------------------===//

cv.subrotmj t0, t1, t2
# CHECK-INSTR: cv.subrotmj t0, t1, t2
# CHECK-ENCODING: [0xfb,0x02,0x73,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj t3, t4, t5
# CHECK-INSTR: cv.subrotmj t3, t4, t5
# CHECK-ENCODING: [0x7b,0x8e,0xee,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj a0, a1, a2
# CHECK-INSTR: cv.subrotmj a0, a1, a2
# CHECK-ENCODING: [0x7b,0x85,0xc5,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj s0, s1, s2
# CHECK-INSTR: cv.subrotmj s0, s1, s2
# CHECK-ENCODING: [0x7b,0x84,0x24,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.subrotmj.div2
//===----------------------------------------------------------------------===//

cv.subrotmj.div2 t0, t1, t2
# CHECK-INSTR: cv.subrotmj.div2 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x22,0x73,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div2 t3, t4, t5
# CHECK-INSTR: cv.subrotmj.div2 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xae,0xee,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div2 a0, a1, a2
# CHECK-INSTR: cv.subrotmj.div2 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xa5,0xc5,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div2 s0, s1, s2
# CHECK-INSTR: cv.subrotmj.div2 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xa4,0x24,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.subrotmj.div4
//===----------------------------------------------------------------------===//

cv.subrotmj.div4 t0, t1, t2
# CHECK-INSTR: cv.subrotmj.div4 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div4 t3, t4, t5
# CHECK-INSTR: cv.subrotmj.div4 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div4 a0, a1, a2
# CHECK-INSTR: cv.subrotmj.div4 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div4 s0, s1, s2
# CHECK-INSTR: cv.subrotmj.div4 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.subrotmj.div8
//===----------------------------------------------------------------------===//

cv.subrotmj.div8 t0, t1, t2
# CHECK-INSTR: cv.subrotmj.div8 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x62,0x73,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div8 t3, t4, t5
# CHECK-INSTR: cv.subrotmj.div8 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xee,0xee,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div8 a0, a1, a2
# CHECK-INSTR: cv.subrotmj.div8 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xe5,0xc5,0x64]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.subrotmj.div8 s0, s1, s2
# CHECK-INSTR: cv.subrotmj.div8 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xe4,0x24,0x65]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.add.div2
//===----------------------------------------------------------------------===//

cv.add.div2 t0, t1, t2
# CHECK-INSTR: cv.add.div2 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x22,0x73,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div2 t3, t4, t5
# CHECK-INSTR: cv.add.div2 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xae,0xee,0x6d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div2 a0, a1, a2
# CHECK-INSTR: cv.add.div2 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xa5,0xc5,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div2 s0, s1, s2
# CHECK-INSTR: cv.add.div2 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xa4,0x24,0x6d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.add.div4
//===----------------------------------------------------------------------===//

cv.add.div4 t0, t1, t2
# CHECK-INSTR: cv.add.div4 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div4 t3, t4, t5
# CHECK-INSTR: cv.add.div4 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x6d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div4 a0, a1, a2
# CHECK-INSTR: cv.add.div4 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div4 s0, s1, s2
# CHECK-INSTR: cv.add.div4 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x6d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.add.div8
//===----------------------------------------------------------------------===//

cv.add.div8 t0, t1, t2
# CHECK-INSTR: cv.add.div8 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x62,0x73,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div8 t3, t4, t5
# CHECK-INSTR: cv.add.div8 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xee,0xee,0x6d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div8 a0, a1, a2
# CHECK-INSTR: cv.add.div8 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xe5,0xc5,0x6c]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.add.div8 s0, s1, s2
# CHECK-INSTR: cv.add.div8 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xe4,0x24,0x6d]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.div2
//===----------------------------------------------------------------------===//

cv.sub.div2 t0, t1, t2
# CHECK-INSTR: cv.sub.div2 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x22,0x73,0x74]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div2 t3, t4, t5
# CHECK-INSTR: cv.sub.div2 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xae,0xee,0x75]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div2 a0, a1, a2
# CHECK-INSTR: cv.sub.div2 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xa5,0xc5,0x74]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div2 s0, s1, s2
# CHECK-INSTR: cv.sub.div2 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xa4,0x24,0x75]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.div4
//===----------------------------------------------------------------------===//

cv.sub.div4 t0, t1, t2
# CHECK-INSTR: cv.sub.div4 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x42,0x73,0x74]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div4 t3, t4, t5
# CHECK-INSTR: cv.sub.div4 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xce,0xee,0x75]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div4 a0, a1, a2
# CHECK-INSTR: cv.sub.div4 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xc5,0xc5,0x74]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div4 s0, s1, s2
# CHECK-INSTR: cv.sub.div4 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xc4,0x24,0x75]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

//===----------------------------------------------------------------------===//
// cv.sub.div8
//===----------------------------------------------------------------------===//

cv.sub.div8 t0, t1, t2
# CHECK-INSTR: cv.sub.div8 t0, t1, t2
# CHECK-ENCODING: [0xfb,0x62,0x73,0x74]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div8 t3, t4, t5
# CHECK-INSTR: cv.sub.div8 t3, t4, t5
# CHECK-ENCODING: [0x7b,0xee,0xee,0x75]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div8 a0, a1, a2
# CHECK-INSTR: cv.sub.div8 a0, a1, a2
# CHECK-ENCODING: [0x7b,0xe5,0xc5,0x74]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

cv.sub.div8 s0, s1, s2
# CHECK-INSTR: cv.sub.div8 s0, s1, s2
# CHECK-ENCODING: [0x7b,0xe4,0x24,0x75]
# CHECK-NO-EXT: instruction requires the following: 'XCVsimd' (CORE-V SIMD ALU){{$}}

