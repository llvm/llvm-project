//=- SPIRVCombine.td - Define SPIRV Combine Rules -------------*-tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

include "llvm/Target/GlobalISel/Combine.td"


def vector_length_sub_to_distance_lowering : GICombineRule <
  (defs root:$root),
  (match (wip_match_opcode G_INTRINSIC):$root,
          [{ return Helper.matchLengthToDistance(*${root}); }]),
  (apply [{ Helper.applySPIRVDistance(*${root}); }])
>;

def vector_select_to_faceforward_lowering : GICombineRule <
  (defs root:$root),
  (match (wip_match_opcode G_SELECT):$root,
          [{ return Helper.matchSelectToFaceForward(*${root}); }]),
  (apply [{ Helper.applySPIRVFaceForward(*${root}); }])
>;

def matrix_transpose_lowering
    : GICombineRule<(defs root:$root),
                    (match (wip_match_opcode G_INTRINSIC):$root,
                        [{ return Helper.matchMatrixTranspose(*${root}); }]),
                    (apply [{ Helper.applyMatrixTranspose(*${root}); }])>;

def matrix_multiply_lowering
    : GICombineRule<(defs root:$root),
                    (match (wip_match_opcode G_INTRINSIC):$root,
                        [{ return Helper.matchMatrixMultiply(*${root}); }]),
                    (apply [{ Helper.applyMatrixMultiply(*${root}); }])>;

def SPIRVPreLegalizerCombiner
    : GICombiner<"SPIRVPreLegalizerCombinerImpl",
                 [vector_length_sub_to_distance_lowering,
                  vector_select_to_faceforward_lowering,
                  matrix_transpose_lowering, matrix_multiply_lowering]> {
  let CombineAllMethodName = "tryCombineAllImpl";
}
