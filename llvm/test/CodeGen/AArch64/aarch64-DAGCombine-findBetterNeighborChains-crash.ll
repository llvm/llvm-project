; RUN: llc < %s -march=arm64
; Make sure we are not crashing on this test.

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

declare void @extern(i8*)

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i64(i8* nocapture, i8, i64, i32, i1) #0

; Function Attrs: nounwind
define void @func(float* noalias %arg, i32* noalias %arg1, i8* noalias %arg2, i8* noalias %arg3) #1 {
bb:
  %tmp = getelementptr inbounds i8, i8* %arg2, i64 88
  tail call void @llvm.memset.p0i8.i64(i8* noalias %arg2, i8 0, i64 40, i32 8, i1 false)
  store i8 0, i8* %arg3
  store i8 2, i8* %arg2
  store float 0.000000e+00, float* %arg
  %tmp4 = bitcast i8* %tmp to <4 x float>*
  store volatile <4 x float> zeroinitializer, <4 x float>* %tmp4
  store i32 5, i32* %arg1
  tail call void @extern(i8* %tmp)
  ret void
}

; Function Attrs: nounwind
define void @func2(float* noalias %arg, i32* noalias %arg1, i8* noalias %arg2, i8* noalias %arg3) #1 {
bb:
  %tmp = getelementptr inbounds i8, i8* %arg2, i64 88
  tail call void @llvm.memset.p0i8.i64(i8* noalias %arg2, i8 0, i64 40, i32 8, i1 false)
  store i8 0, i8* %arg3
  store i8 2, i8* %arg2
  store float 0.000000e+00, float* %arg
  %tmp4 = bitcast i8* %tmp to <4 x float>*
  store <4 x float> zeroinitializer, <4 x float>* %tmp4
  store i32 5, i32* %arg1
  tail call void @extern(i8* %tmp)
  ret void
}

attributes #0 = { argmemonly nounwind }
attributes #1 = { nounwind "target-cpu"="cortex-a53" }
