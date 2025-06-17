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
          [{ return matchLengthToDistance(*${root}, MRI); }]),
  (apply [{ applySPIRVDistance(*${root}, MRI, B); }])
>;

def SPIRVPreLegalizerCombiner
    : GICombiner<"SPIRVPreLegalizerCombinerImpl",
                       [vector_length_sub_to_distance_lowering]> {
    let CombineAllMethodName = "tryCombineAllImpl";
}
