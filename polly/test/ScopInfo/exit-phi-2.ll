; RUN: opt %loadPolly -polly-stmt-granularity=bb -polly-print-scops -disable-output < %s | FileCheck %s
;
; Check that there is no MK_ExitPHI READ access.
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @local_book_besterror() {
entry:
  %0 = load i64, ptr undef, align 8
  %conv = trunc i64 %0 to i32
  br label %for.body64

for.body64:
  %bestf.011 = phi float [ 0.000000e+00, %entry ], [ %this.0.bestf.0, %if.end92 ]
  br label %for.body74

for.body74:
  br i1 false, label %for.body74, label %for.cond71.for.end85_crit_edge

for.cond71.for.end85_crit_edge:
  %cmp88 = fcmp olt float undef, %bestf.011
  %this.0.bestf.0 = select i1 undef, float undef, float %bestf.011
  br label %if.end92

if.end92:
  br i1 undef, label %for.body64, label %for.cond60.if.end96.loopexit_crit_edge

for.cond60.if.end96.loopexit_crit_edge:           ; preds = %if.end92
  ret void
}

; CHECK:      Statements {
; CHECK-NEXT:     Stmt_for_cond71_for_end85_crit_edge
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             { Stmt_for_cond71_for_end85_crit_edge[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             { Stmt_for_cond71_for_end85_crit_edge[] -> [] };
; CHECK-NEXT:         ReadAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_for_cond71_for_end85_crit_edge[] -> MemRef_bestf_011[] };
; CHECK-NEXT:         MustWriteAccess :=    [Reduction Type: NONE] [Scalar: 1]
; CHECK-NEXT:             { Stmt_for_cond71_for_end85_crit_edge[] -> MemRef_this_0_bestf_0[] };
; CHECK-NEXT: }
