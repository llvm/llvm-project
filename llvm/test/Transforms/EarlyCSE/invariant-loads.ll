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

; By assumption, the call can't change contents of p
; LangRef is a bit unclear about whether the store is reachable, so
; for the moment we chose to be conservative and just assume it's valid
; to restore the same unchanging value.
define void @test_dse1(i32* %p) {
; CHECK-LABEL: @test_dse1
; CHECK-NOT: store
  %v1 = load i32, i32* %p, !invariant.load !{}
  call void @clobber_and_use(i32 %v1)
  store i32 %v1, i32* %p
  ret void
}

; By assumption, v1 must equal v2 (TODO)
define void @test_false_negative_dse2(i32* %p, i32 %v2) {
; CHECK-LABEL: @test_false_negative_dse2
; CHECK: store
  %v1 = load i32, i32* %p, !invariant.load !{}
  call void @clobber_and_use(i32 %v1)
  store i32 %v2, i32* %p
  ret void
}

; If we remove the load, we still start an invariant scope since
; it lets us remove later loads not explicitly marked invariant
define void @test_scope_start_without_load(i32* %p) {
; CHECK-LABEL: @test_scope_start_without_load
; CHECK:   %v1 = load i32, i32* %p
; CHECK:   %add = add i32 %v1, %v1
; CHECK:   call void @clobber_and_use(i32 %add)
; CHECK:   call void @clobber_and_use(i32 %v1)
; CHECK:   ret void
  %v1 = load i32, i32* %p
  %v2 = load i32, i32* %p, !invariant.load !{}
  %add = add i32 %v1, %v2
  call void @clobber_and_use(i32 %add)
  %v3 = load i32, i32* %p
  call void @clobber_and_use(i32 %v3)
  ret void
}

; If we already have an invariant scope, don't want to start a new one
; with a potentially greater generation.  This hides the earlier invariant
; load
define void @test_scope_restart(i32* %p) {
; CHECK-LABEL: @test_scope_restart
; CHECK:   %v1 = load i32, i32* %p
; CHECK:   call void @clobber_and_use(i32 %v1)
; CHECK:   %add = add i32 %v1, %v1
; CHECK:   call void @clobber_and_use(i32 %add)
; CHECK:   call void @clobber_and_use(i32 %v1)
; CHECK:   ret void
  %v1 = load i32, i32* %p, !invariant.load !{}
  call void @clobber_and_use(i32 %v1)
  %v2 = load i32, i32* %p, !invariant.load !{}
  %add = add i32 %v1, %v2
  call void @clobber_and_use(i32 %add)
  %v3 = load i32, i32* %p
  call void @clobber_and_use(i32 %v3)
  ret void
}
