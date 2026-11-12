; RUN: llc < %s -mtriple=aarch64-unknown-linux-gnu
; Make sure we are not crashing on this test.

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

declare void @extern(ptr)

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0.i64(ptr nocapture, i8, i64, i1) #0

; Function Attrs: nounwind
define void @func(ptr noalias %arg, ptr noalias %arg1, ptr noalias %arg2, ptr noalias %arg3) #1 {
bb:
  %tmp = getelementptr inbounds i8, ptr %arg2, i64 88
  tail call void @llvm.memset.p0.i64(ptr align 8 noalias %arg2, i8 0, i64 40, i1 false)
  store i8 0, ptr %arg3
  store i8 2, ptr %arg2
  store float 0.000000e+00, ptr %arg
  store volatile <4 x float> zeroinitializer, ptr %tmp
  store i32 5, ptr %arg1
  tail call void @extern(ptr %tmp)
  ret void
}

; Function Attrs: nounwind
define void @func2(ptr noalias %arg, ptr noalias %arg1, ptr noalias %arg2, ptr noalias %arg3) #1 {
bb:
  %tmp = getelementptr inbounds i8, ptr %arg2, i64 88
  tail call void @llvm.memset.p0.i64(ptr align 8 noalias %arg2, i8 0, i64 40, i1 false)
  store i8 0, ptr %arg3
  store i8 2, ptr %arg2
  store float 0.000000e+00, ptr %arg
  store <4 x float> zeroinitializer, ptr %tmp
  store i32 5, ptr %arg1
  tail call void @extern(ptr %tmp)
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind "target-cpu"="cortex-a53" }
