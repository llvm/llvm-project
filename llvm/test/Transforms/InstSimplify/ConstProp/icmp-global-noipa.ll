; RUN: opt < %s -passes=instsimplify -S | FileCheck %s

; noipa doesn't affect which definition (and so which address) a function
; resolves to, so comparisons of noipa function addresses fold just like those
; of other functions with the same linkage.

define void @f() {
  ret void
}

define void @g() {
  ret void
}

define void @noipa_f() noipa {
  ret void
}

define weak void @weak_f() {
  ret void
}

define i1 @plain_eq() {
; CHECK-LABEL: @plain_eq(
; CHECK-NEXT:    ret i1 false
  %cmp = icmp eq ptr @f, @g
  ret i1 %cmp
}

define i1 @noipa_eq() {
; CHECK-LABEL: @noipa_eq(
; CHECK-NEXT:    ret i1 false
  %cmp = icmp eq ptr @noipa_f, @g
  ret i1 %cmp
}

; Interposable functions may resolve to the same address, so don't fold.
define i1 @weak_eq() {
; CHECK-LABEL: @weak_eq(
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq ptr @weak_f, @g
; CHECK-NEXT:    ret i1 [[CMP]]
  %cmp = icmp eq ptr @weak_f, @g
  ret i1 %cmp
}
