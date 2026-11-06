//=- SPIRVCombine.td - Define SPIRV Combine Rules -------------*-tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

include "llvm/Target/GlobalISel/Combine.td"


// length(x - y) -> distance(x, y)
def vector_length_sub_to_distance_lowering : GICombineRule <
  (defs root:$root),
  (match (G_FSUB $sub, $x, $y),
         (int_spv_length $dst, $sub):$root),
  (apply (int_spv_distance $dst, $x, $y))
>;

def vector_fdiv_length_to_normalize_lowering : GICombineRule <
  (defs root:$root),
  (match (G_FDIV $dst, $src1, $src2):$root,
          [{ return Helper.matchFDivToNormalize(*${root}); }]),
  (apply [{ Helper.applySPIRVNormalize(*${root}); }])
>;

def vector_select_to_faceforward_lowering : GICombineRule <
  (defs root:$root),
  (match (G_SELECT $dst, $cond, $t, $f):$root,
          [{ return Helper.matchSelectToFaceForward(*${root}); }]),
  (apply [{ Helper.applySPIRVFaceForward(*${root}); }])
>;

def matrix_transpose_lowering
    : GICombineRule<
          (defs root:$root),
          (match (int_matrix_transpose $dst, $matrix, $rows, $cols):$root),
          (apply [{ Helper.applyMatrixTranspose(*${root}); }])>;

def matrix_multiply_lowering
    : GICombineRule<
          (defs root:$root),
          (match (int_matrix_multiply $dst, $a, $b, $m, $n, $k):$root),
          (apply [{ Helper.applyMatrixMultiply(*${root}); }])>;

def SPIRVPreLegalizerCombiner
    : GICombiner<"SPIRVPreLegalizerCombinerImpl",
                 [vector_length_sub_to_distance_lowering,
                  vector_fdiv_length_to_normalize_lowering,
                  vector_select_to_faceforward_lowering,
                  matrix_transpose_lowering, matrix_multiply_lowering]> {
  let CombineAllMethodName = "tryCombineAllImpl";
}
