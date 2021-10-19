; RUN: opt %loadPolly -polly-optree -polly-opt-isl -polly-codegen -S < %s | FileCheck %s
;
; llvm.org/PR35362
; isl codegen does not allow to generate isl_ast_expr from pw_aff which have an
; empty domain. This happens in this case because the pw_aff's domain is
; excluded by the SCoP's parameter context.

target datalayout = "e-m:w-i64:64-f80:128-n8:16:32:64-S128"

@c = external local_unnamed_addr global i8
@a = external local_unnamed_addr global i16
@b = external local_unnamed_addr global i8

define void @fn1() {
entry:
  %a.promoted = load i16, i16* @a
  br label %for.cond

for.cond:                                         ; preds = %for.cond3.for.end_crit_edge, %entry
  %inc.lcssa17 = phi i16 [ 0, %for.cond3.for.end_crit_edge ], [ %a.promoted, %entry ]
  br label %for.body

for.body:                                         ; preds = %for.cond
  %conv = zext i16 %inc.lcssa17 to i32
  %div = udiv i32 -286702568, %conv
  br i1 undef, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  unreachable

if.end:                                           ; preds = %for.body
  br label %for.body5.lr.ph

for.body5.lr.ph:                                  ; preds = %if.end
  %tmp = load i8, i8* @b, align 1
  %cmp = icmp eq i32 %div, 1
  br i1 %cmp, label %for.body5.lr.ph.split.us, label %for.body5.lr.ph.split

for.body5.lr.ph.split.us:                         ; preds = %for.body5.lr.ph
  br label %lor.end.us.peel

lor.end.us.peel:                                  ; preds = %for.body5.lr.ph.split.us
  %inc.us.peel = add i16 %inc.lcssa17, 1
  br i1 false, label %for.cond3.for.end_crit_edge, label %for.body5.us.peel.next

for.body5.us.peel.next:                           ; preds = %lor.end.us.peel
  br label %lor.end.us

lor.end.us:                                       ; preds = %lor.end.us, %for.body5.us.peel.next
  %tmp1 = phi i16 [ %inc.us.peel, %for.body5.us.peel.next ], [ %inc.us, %lor.end.us ]
  %inc.us = add i16 %tmp1, 1
  %tobool4.us = icmp eq i16 %inc.us, 0
  br i1 %tobool4.us, label %for.cond3.for.end_crit_edge, label %lor.end.us

for.body5.lr.ph.split:                            ; preds = %for.body5.lr.ph
  br label %lor.end.peel

lor.end.peel:                                     ; preds = %for.body5.lr.ph.split
  %inc.peel = add i16 %inc.lcssa17, 1
  br i1 false, label %for.cond3.for.end_crit_edge, label %for.body5.peel.next

for.body5.peel.next:                              ; preds = %lor.end.peel
  br label %lor.end

lor.end:                                          ; preds = %lor.end, %for.body5.peel.next
  %tmp2 = phi i16 [ %inc.peel, %for.body5.peel.next ], [ %inc, %lor.end ]
  %inc = add i16 %tmp2, 1
  %tobool4 = icmp eq i16 %inc, 0
  br i1 %tobool4, label %for.cond3.for.end_crit_edge, label %lor.end

for.cond3.for.end_crit_edge:                      ; preds = %lor.end, %lor.end.peel, %lor.end.us, %lor.end.us.peel
  %tmp3 = phi i8 [ %tmp, %lor.end.us.peel ], [ %tmp, %lor.end.peel ], [ %tmp, %lor.end.us ], [ %tmp, %lor.end ]
  store i8 4, i8* @c
  br label %for.cond
}


; The reference to @b should have been generated from an isl_ast_expr.
; Because isl is unable to generate it in this case, the code generator
; resorted to use the pointer argument of %tmp = load ... .
; It is not important since this code will never be executed.

; CHECK:      polly.stmt.lor.end.us.peel:
; CHECK-NEXT:   %tmp_p_scalar_2 = load i8, i8* @b
; CHECK-NEXT:   store i8 %tmp_p_scalar_2, i8* %tmp3.phiops
; CHECK-NEXT:   br label %polly.merge
