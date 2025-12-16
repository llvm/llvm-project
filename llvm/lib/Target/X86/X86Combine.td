//===---------------------- X86Combiner.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

def X86PostLegalizerCombiner : GICombiner<"X86PostLegalizerCombinerImpl", [redundant_or, constant_fold_fp_ops, identity_combines, copy_prop]> {
    let CombineAllMethodName = "tryCombineAllImpl";
}