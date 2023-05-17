; RUN: opt -S --passes=ipsccp,deadargelim -funcspec-for-literal-constant --force-specialization < %s | FileCheck %s

; Test that all of `f0`, `f1`, and `f2` are specialised, even though `f0` has its address taken
; and `f1` is with external linkage (`f2` was specialised anyway).

@p = global ptr @f0

; `f0` is kept even though all apparent calls are specialized
; CHECK-LABEL: define internal i32 @f0(
define internal i32 @f0(i32 %i) {
  %v = add i32 %i, 1
  ret i32 %v
}

; Likewise, `f1` is kept, because of the external linkage
; CHECK-LABEL: define i32 @f1(
define i32 @f1(i32 %i) {
  %v = add i32 %i, 1
  ret i32 %v
}

; `f2` is fully specialised.
; CHECK-NOT: defined internal i32 @f2()
define internal i32 @f2(i32 %i) {
  %v = add i32 %i, 1
  ret i32 %v
}

;; All calls are to specilisation instances.

; CHECK-LABEL: define i32 @g0
; CHECK:         call void @f0.[[#A:]]()
; CHECK-NEXT:    call void @f1.[[#B:]]()
; CHECK-NEXT:    call void @f2.[[#C:]]()
; CHECK-NEXT:    ret i32 9
define i32 @g0(i32 %i) {
  %u0 = call i32 @f0(i32 1)
  %u1 = call i32 @f1(i32 2)
  %u2 = call i32 @f2(i32 3)
  %v0 = add i32 %u0, %u1
  %v = add i32 %v0, %u2
  ret i32 %v
}

; CHECK-LABEL: define i32 @g1
; CHECK:         call void @f0.[[#D:]]()
; CHECK-NEXT:    call void @f1.[[#E:]]()
; CHECK-NEXT:    call void @f2.[[#F:]]()
; CHECK-NEXT:    ret i32 12
define i32 @g1(i32 %i) {
  %u0 = call i32 @f0(i32 2)
  %u1 = call i32 @f1(i32 3)
  %u2 = call i32 @f2(i32 4)
  %v0 = add i32 %u0, %u1
  %v = add i32 %v0, %u2
  ret i32 %v
}

; All of the function are specialized and all clones are with internal linkage.

; CHECK-DAG: define internal void @f0.[[#A]]() {
; CHECK-DAG: define internal void @f1.[[#B]]() {
; CHECK-DAG: define internal void @f2.[[#C]]() {
; CHECK-DAG: define internal void @f0.[[#D]]() {
; CHECK-DAG: define internal void @f1.[[#E]]() {
; CHECK-DAG: define internal void @f2.[[#F]]() {
