; RUN: opt -S -indvars < %s | FileCheck %s

; Check that this test does not crash.

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.10.0"

define void @re_update_line(i8* %new) {
; CHECK: @re_update_line(
entry:
  %incdec.ptr6 = getelementptr inbounds i8, i8* %new, i64 1
  br label %for.cond.11.preheader

for.cond.11.preheader:                            ; preds = %for.inc.26, %entry
  %n.154 = phi i8* [ %new, %entry ], [ %incdec.ptr27, %for.inc.26 ]
  %cmp12.52 = icmp ult i8* %n.154, %incdec.ptr6
  br i1 %cmp12.52, label %land.rhs.16.lr.ph, label %for.inc.26

land.rhs.16.lr.ph:                                ; preds = %for.cond.11.preheader
  br label %land.rhs.16

for.cond.11:                                      ; preds = %land.rhs.16
  %incdec.ptr24 = getelementptr inbounds i8, i8* %p.053, i64 1
  %cmp12 = icmp ult i8* %p.053, %new
  br i1 %cmp12, label %land.rhs.16, label %for.inc.26

land.rhs.16:                                      ; preds = %for.cond.11, %land.rhs.16.lr.ph
  %p.053 = phi i8* [ %n.154, %land.rhs.16.lr.ph ], [ %incdec.ptr24, %for.cond.11 ]
  br i1 undef, label %for.cond.11, label %for.inc.26

for.inc.26:                                       ; preds = %land.rhs.16, %for.cond.11, %for.cond.11.preheader
  %incdec.ptr27 = getelementptr inbounds i8, i8* %n.154, i64 1
  br i1 false, label %for.cond.11.preheader, label %for.end.28

for.end.28:                                       ; preds = %for.inc.26
  ret void
}
