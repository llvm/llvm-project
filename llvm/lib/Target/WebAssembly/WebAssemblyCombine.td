//=- WebAssemblyCombine.td - Define Wasm Combine Rules -------*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//
//===----------------------------------------------------------------------===//

include "llvm/Target/GlobalISel/Combine.td"

def WebAssemblyPreLegalizerCombiner: GICombiner<
  "WebAssemblyPreLegalizerCombinerImpl", []> {
}

// Post-legalization combines which are primarily optimizations.
def WebAssemblyPostLegalizerCombiner
    : GICombiner<"WebAssemblyPostLegalizerCombinerImpl",
                 []> {
}
