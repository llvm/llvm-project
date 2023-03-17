; PR30213. Check scev expand will generate correct code if the value
; in ValueOffsetPair is of pointer type.
; RUN: opt -mtriple=i386-apple-macosx10.12.0 < %s -loop-reduce -S | FileCheck %s

; CHECK: %ptr4.ptr1 = select i1 %cmp.i, ptr %ptr4, ptr %ptr1
; CHECK-NEXT: %scevgep = getelementptr i8, ptr %ptr4.ptr1, i32 1
; CHECK-NEXT: br label %while.cond.i

target datalayout = "e-m:o-p:32:32-f64:32:64-f80:128-n8:16:32-S128"
target triple = "i386-apple-macosx10.12.0"

; Function Attrs: nounwind optsize ssp
define void @Foo() {
entry:
  switch i2 undef, label %sw.epilog102 [
    i2 -2, label %sw.bb28
  ]

sw.bb28:                                          ; preds = %entry
  %0 = load ptr, ptr undef, align 2
  %ptr1 = getelementptr inbounds i8, ptr undef, i32 -1
  %ptr4 = getelementptr inbounds i8, ptr %0, i32 -1
  %cmp.i = icmp ult ptr undef, %0
  %ptr4.ptr1 = select i1 %cmp.i, ptr %ptr4, ptr %ptr1
  br label %while.cond.i

while.cond.i:                                     ; preds = %while.cond.i, %sw.bb28
  %currPtr.1.i = phi ptr [ %incdec.ptr.i, %while.cond.i ], [ %ptr4.ptr1, %sw.bb28 ]
  %incdec.ptr.i = getelementptr inbounds i8, ptr %currPtr.1.i, i32 1
  %1 = load i8, ptr %incdec.ptr.i, align 1
  br label %while.cond.i

sw.epilog102:                                     ; preds = %entry
  unreachable
}

