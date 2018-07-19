; RUN: opt -S -early-cse < %s | FileCheck %s
; RUN: opt -S -basicaa -early-cse-memssa < %s | FileCheck %s

declare void @clobber_and_use(i32)

define void @f_0(i32* %ptr) {
; CHECK-LABEL: @f_0(
; CHECK:   %val0 = load i32, i32* %ptr, !invariant.load !0
; CHECK:   call void @clobber_and_use(i32 %val0)
; CHECK:   call void @clobber_and_use(i32 %val0)
; CHECK:   call void @clobber_and_use(i32 %val0)
; CHECK:   ret void

  %val0 = load i32, i32* %ptr, !invariant.load !{}
  call void @clobber_and_use(i32 %val0)
  %val1 = load i32, i32* %ptr, !invariant.load !{}
  call void @clobber_and_use(i32 %val1)
  %val2 = load i32, i32* %ptr, !invariant.load !{}
  call void @clobber_and_use(i32 %val2)
  ret void
}

define void @f_1(i32* %ptr) {
; We can forward invariant loads to non-invariant loads.

; CHECK-LABEL: @f_1(
; CHECK:   %val0 = load i32, i32* %ptr, !invariant.load !0
; CHECK:   call void @clobber_and_use(i32 %val0)
; CHECK:   call void @clobber_and_use(i32 %val0)

  %val0 = load i32, i32* %ptr, !invariant.load !{}
  call void @clobber_and_use(i32 %val0)
  %val1 = load i32, i32* %ptr
  call void @clobber_and_use(i32 %val1)
  ret void
}

define void @f_2(i32* %ptr) {
; We can forward a non-invariant load into an invariant load.

; CHECK-LABEL: @f_2(
; CHECK:   %val0 = load i32, i32* %ptr
; CHECK:   call void @clobber_and_use(i32 %val0)
; CHECK:   call void @clobber_and_use(i32 %val0)

  %val0 = load i32, i32* %ptr
  call void @clobber_and_use(i32 %val0)
  %val1 = load i32, i32* %ptr, !invariant.load !{}
  call void @clobber_and_use(i32 %val1)
  ret void
}

define void @f_3(i1 %cond, i32* %ptr) {
; CHECK-LABEL: @f_3(
  %val0 = load i32, i32* %ptr, !invariant.load !{}
  call void @clobber_and_use(i32 %val0)
  br i1 %cond, label %left, label %right

; CHECK:  %val0 = load i32, i32* %ptr, !invariant.load !0
; CHECK: left:
; CHECK-NEXT:  call void @clobber_and_use(i32 %val0)

left:
  %val1 = load i32, i32* %ptr
  call void @clobber_and_use(i32 %val1)
  ret void

right:
  ret void
}

define void @f_4(i1 %cond, i32* %ptr) {
; Negative test -- can't forward %val0 to %va1 because that'll break
; def-dominates-use.

; CHECK-LABEL: @f_4(
  br i1 %cond, label %left, label %merge

left:
; CHECK: left:
; CHECK-NEXT:  %val0 = load i32, i32* %ptr, !invariant.load !
; CHECK-NEXT:  call void @clobber_and_use(i32 %val0)

  %val0 = load i32, i32* %ptr, !invariant.load !{}
  call void @clobber_and_use(i32 %val0)
  br label %merge

merge:
; CHECK: merge:
; CHECK-NEXT:   %val1 = load i32, i32* %ptr
; CHECK-NEXT:   call void @clobber_and_use(i32 %val1)

  %val1 = load i32, i32* %ptr
  call void @clobber_and_use(i32 %val1)
  ret void
}
