; RUN: opt < %s -loop-interchange -S | FileCheck %s

; BB latch1 is the loop latch, but does not exit the loop.
define void @foo() {
entry:
  %dest = alloca i16*, align 8
  br label %header1

header1:
  %0 = phi i16* [ %2, %latch1 ], [ undef, %entry ]
  br i1 false, label %inner, label %loopexit

inner:
  br i1 undef, label %inner.ph, label %latch1

inner.ph:
  br label %inner.body

inner.body:
  %1 = load i16, i16* %0, align 2
  store i16* inttoptr (i64 2 to i16*), i16** %dest, align 8
  br i1 false, label %inner.body, label %inner.loopexit

inner.loopexit:
  br label %latch1

latch1:
  %2 = phi i16* [ %0, %inner ], [ undef, %inner.loopexit ]
  br label %header1

loopexit:                                         ; preds = %header1
  unreachable
}

; CHECK-LABEL: inner.body:
; CHECK: br i1 false, label %inner.body, label %inner.loopexit
; CHECK: latch1:
; CHECK-NEXT: %2 = phi i16* [ %0, %inner ], [ undef, %inner.loopexit ]
; CHECK-NEXT: br label %header1

