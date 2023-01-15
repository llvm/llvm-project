; RUN: opt %s -O2 -S | FileCheck %s

; Check that llvm.x86.avx2.gather.d.ps.256 intrinsic is not eliminated as gather and store memory accesses are based on arr.ptr
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readonly
declare <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float>, ptr, <8 x i32>, <8 x float>, i8) #0

; Function Attrs: nounwind
define <8 x float> @foo1(ptr noalias %arr.ptr, ptr noalias readonly %vix.ptr, ptr noalias %t2.ptr) #1 {
allocas:
  %vix = load <8 x i32>, ptr %vix.ptr, align 4
  %t1.ptr = getelementptr i8, ptr %arr.ptr, i8 4
  
  %v1 = tail call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, ptr %arr.ptr, <8 x i32> %vix, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, i8 1) #2
  store i8 1, ptr %t1.ptr, align 4

  %v2 = tail call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, ptr %arr.ptr, <8 x i32> %vix, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, i8 1) #2
  %res = fadd <8 x float> %v1, %v2

  ret <8 x float> %res
}
; CHECK: foo1
; CHECK: llvm.x86.avx2.gather.d.ps.256
; CHECK: store
; CHECK: llvm.x86.avx2.gather.d.ps.256

; Check that second gather is eliminated as gather and store memory accesses are based on different no-aliasing pointers

; Function Attrs: nounwind
define <8 x float> @foo2(ptr noalias readonly %arr.ptr, ptr noalias readonly %vix.ptr, ptr noalias %t2.ptr) #1 {
allocas:
  %vix = load <8 x i32>, ptr %vix.ptr, align 4
  %t1.ptr = getelementptr i8, ptr %arr.ptr, i8 4
  
  %v1 = tail call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, ptr %arr.ptr, <8 x i32> %vix, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, i8 1) #2
  store i8 1, ptr %t2.ptr, align 4

  %v2 = tail call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, ptr %arr.ptr, <8 x i32> %vix, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, i8 1) #2
  %res = fadd <8 x float> %v1, %v2

  ret <8 x float> %res
}
; CHECK: foo2
; CHECK: llvm.x86.avx2.gather.d.ps.256
; CHECK: store
; CHECK-NOT: llvm.x86.avx2.gather.d.ps.256

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind "target-cpu"="corei7-avx" "target-features"="+avx2,+popcnt,+cmov,+f16c,+rdrnd,+fma" }
attributes #2 = { nounwind }

