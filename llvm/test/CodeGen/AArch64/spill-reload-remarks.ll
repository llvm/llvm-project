; RUN: llc -mtriple=aarch64-linux-gnu -verify-machineinstrs -mattr=+sme -pass-remarks-missed='regalloc' %s -o - 2>&1 | FileCheck %s

; We should have both spill and reload for %arg.

; CHECK: remark: <unknown>:0:0: 2 spills 2.000000e+00 total spills cost 3 reloads 3.000000e+00 total reloads cost generated in function
define <vscale x 2 x i1> @streaming_compatible_with_predicate_vectors(<vscale x 2 x i1> %arg) "aarch64_pstate_sm_compatible" nounwind #0 {
  %res = call <vscale x 2 x i1> @normal_callee_predicate_vec_arg(<vscale x 2 x i1> %arg)
  %and = and <vscale x 2 x i1> %res, %arg
  ret <vscale x 2 x i1> %and
}

declare <vscale x 2 x i1> @normal_callee_predicate_vec_arg(<vscale x 2 x i1>)

attributes #0 = { nounwind "target-features"="+sve" }
