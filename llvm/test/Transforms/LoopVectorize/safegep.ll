; RUN: opt -S -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=1  < %s |  FileCheck %s
target datalayout = "e-p:32:32:32-S128-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f16:16:16-f32:32:32-f64:32:64-f128:128:128-v64:64:64-v128:128:128-a0:0:64-f80:32:32-n8:16:32"


; We can vectorize this code because if the address computation would wrap then
; a load from 0 would take place which is undefined behaviour in address space 0
; according to LLVM IR semantics.

; PR16592

; CHECK-LABEL: @safe(
; CHECK: <4 x float>

define void @safe(float* %A, float* %B, float %K) {
entry:
  br label %"<bb 3>"

"<bb 3>":
  %i_15 = phi i32 [ 0, %entry ], [ %i_19, %"<bb 3>" ]
  %pp3 = getelementptr float, float* %A, i32 %i_15
  %D.1396_10 = load float, float* %pp3, align 4
  %pp24 = getelementptr float, float* %B, i32 %i_15
  %D.1398_15 = load float, float* %pp24, align 4
  %D.1399_17 = fadd float %D.1398_15, %K
  %D.1400_18 = fmul float %D.1396_10, %D.1399_17
  store float %D.1400_18, float* %pp3, align 4
  %i_19 = add nsw i32 %i_15, 1
  %exitcond = icmp ne i32 %i_19, 64
  br i1 %exitcond, label %"<bb 3>", label %return

return:
  ret void
}

; In a non-default address space we don't have this rule.

; CHECK-LABEL: @notsafe(
; CHECK-NOT: <4 x float>

define void @notsafe(float addrspace(5) * %A, float* %B, float %K) {
entry:
  br label %"<bb 3>"

"<bb 3>":
  %i_15 = phi i32 [ 0, %entry ], [ %i_19, %"<bb 3>" ]
  %pp3 = getelementptr float, float addrspace(5) * %A, i32 %i_15
  %D.1396_10 = load float, float addrspace(5) * %pp3, align 4
  %pp24 = getelementptr float, float* %B, i32 %i_15
  %D.1398_15 = load float, float* %pp24, align 4
  %D.1399_17 = fadd float %D.1398_15, %K
  %D.1400_18 = fmul float %D.1396_10, %D.1399_17
  store float %D.1400_18, float addrspace(5) * %pp3, align 4
  %i_19 = add nsw i32 %i_15, 1
  %exitcond = icmp ne i32 %i_19, 64
  br i1 %exitcond, label %"<bb 3>", label %return

return:
  ret void
}


