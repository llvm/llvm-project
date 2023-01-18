; RUN: opt %loadPolly -polly-print-detect -disable-output < %s | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

define void @remove_all_children(ptr %eclass) {
entry:
  br label %while.body

while.cond.loopexit:                              ; preds = %while.body50, %while.end44
  fence seq_cst
  br label %while.cond.backedge

while.body:                                       ; preds = %while.cond.backedge, %entry
  br label %if.end33

while.cond.backedge:                              ; preds = %while.end30, %while.cond.loopexit
  br i1 false, label %while.body, label %while.end60

if.end33:                                         ; preds = %while.end30
  br i1 false, label %while.body36, label %while.end44

while.body36:                                     ; preds = %while.body36, %while.body36.lr.ph
  %indvar77 = phi i64 [ 0, %if.end33 ], [ %indvar.next78, %while.body36 ]
  %indvar.next78 = add i64 %indvar77, 1
  br i1 false, label %while.body36, label %while.end44

while.end44:                                      ; preds = %while.body36, %if.end33
  br i1 false, label %while.body50, label %while.cond.loopexit

while.body50:                                     ; preds = %while.body50, %while.body50.lr.ph
  %indvar79 = phi i64 [ 0, %while.end44 ], [ %indvar.next80, %while.body50 ]
  store i32 0, ptr %eclass, align 4
  %indvar.next80 = add i64 %indvar79, 1
  br i1 false, label %while.body50, label %while.cond.loopexit

while.end60:                                      ; preds = %while.cond.backedge
  ret void
}
; remove_all_children
; CHECK-NOT: Valid Region
; CHECK: Valid Region for Scop: if.end33 => while.cond.loopexit
; CHECK-NOT: Valid Region
