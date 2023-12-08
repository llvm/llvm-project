; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s -match-full-lines
; Derived from test-suite/MultiSource/Applications/sgefa/blas.c
;
; The exit value of %i.0320 in land.rhs is not computable.
; It is still synthesizable in %if.end13---%for.end170 because
; %i.0320 is fixed within the SCoP and therefore just another parameter.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @snrm2(i32 %n) local_unnamed_addr {
entry:
  br label %land.rhs

land.rhs:                                         ; preds = %while.body, %entry
  %i.0320 = phi i32 [ 0, %entry ], [ %inc, %while.body ]
  br i1 undef, label %while.body, label %if.end13

while.body:                                       ; preds = %land.rhs
  %inc = add nuw nsw i32 %i.0320, 1
  br label %land.rhs

if.end13:                                         ; preds = %land.rhs
  %i.4284 = add nsw i32 %i.0320, 1
  %cmp131285 = icmp slt i32 %i.4284, %n
  br i1 %cmp131285, label %for.body133.lr.ph, label %for.end170

for.body133.lr.ph:                                ; preds = %if.end13
  br label %for.body133

for.body133:                                      ; preds = %for.body133, %for.body133.lr.ph
  %i.4289 = phi i32 [ %i.4284, %for.body133.lr.ph ], [ %i.4, %for.body133 ]
  %xmax.2287 = phi float [ undef, %for.body133.lr.ph ], [ undef, %for.body133 ]
  %i.4 = add nsw i32 %i.4289, 1
  %exitcond = icmp eq i32 %i.4, %n
  br i1 %exitcond, label %for.end170, label %for.body133

for.end170:                                       ; preds = %for.body133, %if.end13
  ret void
}


; CHECK: Region: %if.end13---%for.end170

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_body133_lr_ph
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p_0, n] -> { Stmt_for_body133_lr_ph[] : n >= 2 + p_0 };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p_0, n] -> { Stmt_for_body133_lr_ph[] -> [0, 0] };
; CHECK-NEXT:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0, n] -> { Stmt_for_body133_lr_ph[] -> MemRef_xmax_2287__phi[] };
; CHECK-NEXT:     Stmt_for_body133
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [p_0, n] -> { Stmt_for_body133[i0] : 0 <= i0 <= -2 - p_0 + n };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [p_0, n] -> { Stmt_for_body133[i0] -> [1, i0] };
; CHECK-NEXT:         MustWriteAccess :=  [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0, n] -> { Stmt_for_body133[i0] -> MemRef_xmax_2287__phi[] };
; CHECK-NEXT:         ReadAccess :=       [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             [p_0, n] -> { Stmt_for_body133[i0] -> MemRef_xmax_2287__phi[] };
; CHECK-NEXT: }
