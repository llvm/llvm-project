//===-- HexagonIntrinsicsDerived.td - Derived intrinsics ---*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Multiply 64-bit and use lower result
//
// Optimized with intrinisics accumulates
//
def : Pat <(mul DoubleRegs:$src1, DoubleRegs:$src2),
      (i64
       (COMBINE_rr
        (HEXAGON_M2_maci
         (HEXAGON_M2_maci
          (i32
           (EXTRACT_SUBREG
            (i64
             (MPYU64 (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src1),
                                          subreg_loreg)),
                     (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src2),
                                          subreg_loreg)))),
            subreg_hireg)),
          (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src1), subreg_loreg)),
          (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src2), subreg_hireg))),
         (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src2), subreg_loreg)),
         (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src1), subreg_hireg))),
        (i32
         (EXTRACT_SUBREG
          (i64
           (MPYU64 (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src1), subreg_loreg)),
                   (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src2),
                                        subreg_loreg)))), subreg_loreg))))>;



