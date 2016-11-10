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
       (A2_combinew
        (M2_maci
         (M2_maci
          (i32
           (EXTRACT_SUBREG
            (i64
             (M2_dpmpyuu_s0 (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src1),
                                          isub_lo)),
                     (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src2),
                                          isub_lo)))),
            isub_hi)),
          (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src1), isub_lo)),
          (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src2), isub_hi))),
         (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src2), isub_lo)),
         (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src1), isub_hi))),
        (i32
         (EXTRACT_SUBREG
          (i64
           (M2_dpmpyuu_s0 
             (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src1), isub_lo)),
                   (i32 (EXTRACT_SUBREG (i64 DoubleRegs:$src2),
                                        isub_lo)))), isub_lo))))>;



