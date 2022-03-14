; RUN: opt %loadPolly -polly-print-scops -disable-output < %s | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

declare void @foo()

define i32 @main(i8* %A) nounwind uwtable {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc5, %entry
  %indvar_out = phi i64 [ %indvar_out.next, %for.inc5 ], [ 0, %entry ]
  call void @foo()
  %tmp = add i64 %indvar_out, 2
  %exitcond5 = icmp ne i64 %indvar_out, 1023
  br i1 %exitcond5, label %for.body, label %for.end7

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %indvar = phi i64 [ %indvar.next, %for.inc ], [ 0, %for.body ]
  %exitcond = icmp ne i64 %indvar, 1023
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %tmp1 = add i64 %tmp, %indvar
  %cmp4 = icmp sgt i64 %tmp1, 1000
  br i1 %cmp4, label %if.then, label %if.end

if.then:                                          ; preds = %for.body3
  %arrayidx = getelementptr i8, i8* %A, i64 %indvar
  store i8 5, i8* %arrayidx
  br label %if.end

if.end:                                           ; preds = %if.end.single_exit
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %indvar.next = add i64 %indvar, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc5

for.inc5:                                         ; preds = %for.end
  %indvar_out.next = add i64 %indvar_out, 1
  br label %for.cond

for.end7:                                         ; preds = %for.cond
  ret i32 0
}

; CHECK:      p0: {0,+,1}<%for.cond>
;
; CHECK:      Statements {
; CHECK-NEXT:     Stmt_if_then
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p_0] -> { Stmt_if_then[i0] : i0 >= 0 and 999 - p_0 <= i0 <= 1022 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p_0] -> { Stmt_if_then[i0] -> [0, i0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [p_0] -> { Stmt_if_then[i0] -> MemRef_A[i0] };
; CHECK-NEXT:     Stmt_for_inc5
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p_0] -> { Stmt_for_inc5[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p_0] -> { Stmt_for_inc5[] -> [1, 0] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0] -> { Stmt_for_inc5[] -> MemRef_indvar_out_next[] };
; CHECK-NEXT: }

