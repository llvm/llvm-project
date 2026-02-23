; RUN: llc -mtriple=aarch64-none-linux-gnu -start-before=aarch64-isel %s -o /dev/null
; Regression test for AArch64 compile-time regression, referring to PR #166962.

define fastcc <2 x i64> @_ZN10tensorflow12_GLOBAL__N_125ComputeXWeightsAndIndicesERKNS_17ImageResizerStateEbPNSt3__u6vectorINS0_17WeightsAndIndicesENS4_9allocatorIS6_EEEE(<2 x i64> %0) {
entry:
  %1 = tail call <2 x i64> @llvm.smin.v2i64(<2 x i64> %0, <2 x i64> <i64 -1, i64 0>)
  ret <2 x i64> %1
}

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare <2 x i64> @llvm.smin.v2i64(<2 x i64>, <2 x i64>) #0

attributes #0 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }