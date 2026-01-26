//=- WebAssemblyCombine.td - Define WASM Combine Rules -------*- tablegen -*-=//
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

def fold_global_offset : GICombineRule<
    (defs root:$root, build_fn_matchinfo:$matchinfo),
    (match (G_GLOBAL_VALUE $base, $g),
           (G_CONSTANT $offset, $imm),
           (G_PTR_ADD $root, $base, $offset, (MIFlags NoUWrap)),
    [{ return matchFoldGlobalOffset(${root}, ${g}, ${imm}, ${matchinfo}); }]),
    (apply [{ Helper.applyBuildFnMO(${root}, ${matchinfo}); }])>;

def form_truncstore_matchdata : GIDefMatchData<"Register">;
def form_truncstore : GICombineRule<
  (defs root:$root, form_truncstore_matchdata:$matchinfo),
  (match (G_STORE $src, $addr):$root,
          [{ return matchFormTruncstore(*${root}, MRI, ${matchinfo}); }]),
  (apply [{ applyFormTruncstore(*${root}, MRI, B, Observer, ${matchinfo}); }])
>;

def WebAssemblyPreLegalizerCombiner: GICombiner<
  "WebAssemblyPreLegalizerCombinerImpl", [all_combines]> {
}

def WebAssemblyO0PreLegalizerCombiner: GICombiner<
  "WebAssemblyO0PreLegalizerCombinerImpl", [optnone_combines]> {
}

// Post-legalization combines which are primarily optimizations.
def WebAssemblyPostLegalizerCombiner
    : GICombiner<"WebAssemblyPostLegalizerCombinerImpl",
                 [fold_global_offset, form_truncstore]> {
}
