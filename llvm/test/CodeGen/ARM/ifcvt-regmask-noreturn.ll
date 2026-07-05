; RUN: llc %s -o - -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:o-p:32:32-f64:32:64-v64:32:64-v128:32:128-a:0:32-n32-S32"
target triple = "thumbv7s-apple-ios8.0.0"

@debw = external global ptr, align 4

; This test ensures that the stack_chk call correctly puts implicit uses/defs for the registers
; live across it when if converting.  This will be R0 which is passed to the call to free at the end
; of the function.
; Prior to this change, the stack_chk call (which does not return) would clobber R0 in its regmask,
; leading to verifier errors because the later use of R0 in free() is not live.

; CHECK-LABEL: @test
; CHECK: stack_chk_fail

; Function Attrs: ssp
define void @test(i32 %argc, ptr nocapture readonly %argv, ptr %ptr, i32 %val) #0 {
entry:
  %count.i = alloca [256 x i32], align 4
  %cmp284.i = icmp eq i32 %val, 0
  br i1 %cmp284.i, label %for.end31.i, label %for.body21.i

for.body21.i:                                     ; preds = %entry
  %arrayidx23.i = getelementptr inbounds [256 x i32], ptr %count.i, i32 0, i32 1
  %tmp20 = load i32, ptr %arrayidx23.i, align 4, !tbaa !0
  store i32 %tmp20, ptr %ptr, align 4, !tbaa !0
  br label %for.end31.i

for.end31.i:                                      ; preds = %for.body21.i, %entry
  %tmp21 = load ptr, ptr @debw, align 4, !tbaa !4
  tail call void @free(ptr %tmp21)
  ret void
}

declare void @free(ptr nocapture)

attributes #0 = { ssp "stack-protector-buffer-size"="8" }

!0 = !{!1, !1, i64 0}
!1 = !{!"int", !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5, !5, i64 0}
!5 = !{!"any pointer", !2, i64 0}
