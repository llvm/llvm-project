; RUN: opt -passes='loop(loop-idiom)' < %s -S | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK-LABEL: @strlen(
; CHECK-NOT: call{{.*}} strlen
; CHECK-LABEL: @wcslen(
; CHECK-NOT: call{{.*}} wcslen

define i64 @strlen(ptr %str) {
entry:
  br label %while.cond

while.cond:
  %str.addr.0 = phi ptr [ %str, %entry ], [ %incdec.ptr, %while.cond ]
  %0 = load i8, ptr %str.addr.0, align 1
  %cmp.not = icmp eq i8 %0, 0
  %incdec.ptr = getelementptr i8, ptr %str.addr.0, i64 1
  br i1 %cmp.not, label %while.end, label %while.cond

while.end:
  %sub.ptr.lhs.cast = ptrtoint ptr %str.addr.0 to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %str to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  ret i64 %sub.ptr.sub
}

define i64 @wcslen(ptr %src) {
entry:
  %cmp = icmp eq ptr %src, null
  br i1 %cmp, label %return, label %lor.lhs.false

lor.lhs.false:                                    ; preds = %entry
  %0 = load i32, ptr %src, align 4
  %cmp1 = icmp eq i32 %0, 0
  br i1 %cmp1, label %return, label %while.cond.preheader

while.cond.preheader:                             ; preds = %lor.lhs.false
  br label %while.cond

while.cond:                                       ; preds = %while.cond.preheader, %while.cond
  %src.pn = phi ptr [ %curr.0, %while.cond ], [ %src, %while.cond.preheader ]
  %curr.0 = getelementptr inbounds i8, ptr %src.pn, i64 4
  %1 = load i32, ptr %curr.0, align 4
  %tobool.not = icmp eq i32 %1, 0
  br i1 %tobool.not, label %while.end, label %while.cond

while.end:                                        ; preds = %while.cond
  %curr.0.lcssa = phi ptr [ %curr.0, %while.cond ]
  %sub.ptr.lhs.cast = ptrtoint ptr %curr.0.lcssa to i64
  %sub.ptr.rhs.cast = ptrtoint ptr %src to i64
  %sub.ptr.sub = sub i64 %sub.ptr.lhs.cast, %sub.ptr.rhs.cast
  %sub.ptr.div = ashr exact i64 %sub.ptr.sub, 2
  br label %return

return:                                           ; preds = %entry, %lor.lhs.false, %while.end
  %retval.0 = phi i64 [ %sub.ptr.div, %while.end ], [ 0, %lor.lhs.false ], [ 0, %entry ]
  ret i64 %retval.0
}


!llvm.module.flags = !{!0}
!0 = !{i32 1, !"wchar_size", i32 4}
