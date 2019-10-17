//=- AArch64.td - Define AArch64 Combine Rules ---------------*- tablegen -*-=//
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

def AArch64PreLegalizerCombinerHelper: GICombinerHelper<
  "AArch64GenPreLegalizerCombinerHelper", [all_combines,
                                           elide_br_by_inverting_cond]> {
  let DisableRuleOption = "aarch64prelegalizercombiner-disable-rule";
}
