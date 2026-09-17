; REQUIRES: asserts
; RUN: opt -disable-verify -debug-pass-manager -passes='default<O2>' -force-vector-width="vscale x 4" -S %s 2>&1 | FileCheck %s --check-prefixes=O2_AARCH64_POST_VECTOR

; O2_AARCH64_POST_VECTOR: Running pass: LoopVectorizePass on f (11 instructions)
; O2_AARCH64_POST_VECTOR: Running analysis: ShouldRunExtraVectorPasses on f
; O2_AARCH64_POST_VECTOR: Running pass: AArch64SVEShuffleOptsPass on loop %vector.body in function f
; O2_AARCH64_POST_VECTOR: Running pass: AArch64SVEShuffleOptsPass on loop %loop in function f
; O2_AARCH64_POST_VECTOR: Invalidating analysis: ShouldRunExtraVectorPasses on f

target triple = "aarch64-unknown-linux-gnu"

define i64 @f(i1 %cond, ptr %src, ptr %dst) {
entry:
  br label %loop

loop:
  %i = phi i64 [ 0, %entry ], [ %inc, %loop ]
  %src.i = getelementptr i32, ptr %src, i64 %i
  %src.v = load i32, ptr %src.i
  %add = add i32 %src.v, 10
  %dst.i = getelementptr i32, ptr %dst, i64 %i
  store i32 %add, ptr %dst.i
  %inc = add nuw nsw i64 %i, 1
  %ec = icmp ne i64 %inc, 1000
  br i1 %ec, label %loop, label %exit

exit:
  ret i64 %i
}
