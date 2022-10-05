; RUN: opt < %s -S -basic-aa -licm | FileCheck %s
; RUN: opt -aa-pipeline=basic-aa -passes='require<opt-remark-emit>,loop-mssa(licm)' < %s -S | FileCheck %s
; This fixes PR22460

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-windows-msvc"

@in = internal unnamed_addr global ptr null, align 8
@out = internal unnamed_addr global ptr null, align 8

; CHECK-LABEL: @bar
; CHECK: entry:
; CHECK: load i64, ptr @in
; CHECK: do.body:
; CHECK-NOT: load

define i64 @bar(i32 %N) {
entry:
  br label %do.body

do.body:                                          ; preds = %l2, %entry
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %l2 ]
  %total = phi i64 [ 0, %entry ], [ %next, %l2 ]
  %c = icmp eq i32 %N, 6
  br i1 %c, label %l1, label %do.body.l2_crit_edge

do.body.l2_crit_edge:                             ; preds = %do.body
  %inval.pre = load ptr, ptr @in, align 8
  br label %l2

l1:                                               ; preds = %do.body
  %v1 = load i64, ptr @in, align 8
  store i64 %v1, ptr @out, align 8
  %0 = inttoptr i64 %v1 to ptr
  br label %l2

l2:                                               ; preds = %do.body.l2_crit_edge, %l1
  %inval = phi ptr [ %inval.pre, %do.body.l2_crit_edge ], [ %0, %l1 ]
  %int = ptrtoint ptr %inval to i64
  %next = add i64 %total, %int
  %inc = add nsw i32 %i.0, 1
  %cmp = icmp slt i32 %inc, %N
  br i1 %cmp, label %do.body, label %do.end

do.end:                                           ; preds = %l2
  ret i64 %total
}
