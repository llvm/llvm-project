; RUN: opt %s -gvn -S -enable-gvn-memdep=false | FileCheck %s
; RUN: opt %s -gvn -S -enable-gvn-memdep=true | FileCheck %s

; Check that llvm.x86.avx2.gather.d.ps.256 intrinsic is not eliminated by GVN
; with and without memory dependence info.
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: nounwind readonly
declare <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float>, i8*, <8 x i32>, <8 x float>, i8) #0

; Function Attrs: nounwind
define <8 x float> @foo1(i8* noalias readonly %arr.ptr, <8 x i32>* noalias readonly %vix.ptr, i8* noalias %t2.ptr) #1 {
allocas:
  %vix = load <8 x i32>, <8 x i32>* %vix.ptr, align 4
  %t1.ptr = getelementptr i8, i8* %arr.ptr, i8 4

  %v1 = tail call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, i8* %arr.ptr, <8 x i32> %vix, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, i8 1) #2
  store i8 1, i8* %t1.ptr, align 4

  %v2 = tail call <8 x float> @llvm.x86.avx2.gather.d.ps.256(<8 x float> undef, i8* %arr.ptr, <8 x i32> %vix, <8 x float> <float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000, float 0xFFFFFFFFE0000000>, i8 1) #2
  %res = fadd <8 x float> %v1, %v2

  ret <8 x float> %res
}
; CHECK: foo1
; CHECK: llvm.x86.avx2.gather.d.ps.256
; CHECK: store
; CHECK: llvm.x86.avx2.gather.d.ps.256
