//===---------------------- X86Combiner.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

// all_x86combines is based on generic all_combines, currently x86 gisel does not
// have vector support and few open issue to address which resulted in failure with
// combines. We will introduce more combines gradually.

def all_x86combines : GICombineGroup<[identity_combines, reassocs, 
    const_combines, undef_combines, trivial_combines, simplify_add_to_sub]>;

def X86PreLegalizerCombiner : GICombiner<"X86PreLegalizerCombinerImpl", [all_x86combines]> {
    let CombineAllMethodName = "tryCombineAllImpl";
}
