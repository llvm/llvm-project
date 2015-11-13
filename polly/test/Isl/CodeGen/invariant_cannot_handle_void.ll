; RUN: opt %loadPolly -S -polly-codegen %s | FileCheck %s
;
; The offset of the %tmp1 load wrt. to %buff (62 bytes) is not divisible
; by the type size (i32 = 4 bytes), thus we will bail out.
;
; CHECK-NOT: polly.start
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define void @sudecrypt(i8* %buff) #0 {
entry:
  br i1 undef, label %cleanup, label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %if.end.6, label %if.then.5

if.then.5:                                        ; preds = %if.end
  unreachable

if.end.6:                                         ; preds = %if.end
  %add.ptr = getelementptr inbounds i8, i8* %buff, i64 62
  %tmp = bitcast i8* %add.ptr to i32*
  %tmp1 = load i32, i32* %tmp, align 4, !tbaa !1
  br i1 false, label %if.then.13, label %switch.early.test

switch.early.test:                                ; preds = %if.end.6
  switch i32 0, label %if.end.16 [
    i32 956, label %if.then.13
    i32 520, label %if.then.13
  ]

if.then.13:                                       ; preds = %switch.early.test, %switch.early.test, %if.end.6
  br label %if.end.16

if.end.16:                                        ; preds = %if.then.13, %switch.early.test
  %key.0 = phi i32 [ undef, %if.then.13 ], [ 0, %switch.early.test ]
  br i1 undef, label %if.end.34, label %if.then.19

if.then.19:                                       ; preds = %if.end.16
  unreachable

if.end.34:                                        ; preds = %if.end.16
  unreachable

cleanup:                                          ; preds = %entry
  ret void
}

attributes #0 = { nounwind uwtable "disable-tail-calls"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+mmx,+sse,+sse2" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.ident = !{!0}

!0 = !{!"clang version 3.8.0 (trunk 250010) (llvm/trunk 250018)"}
!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
