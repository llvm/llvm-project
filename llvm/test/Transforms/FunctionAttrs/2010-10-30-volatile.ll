; RUN: opt < %s -passes=function-attrs -S | FileCheck %s
; PR8279

@g = constant i32 1

; CHECK: Function Attrs
; CHECK-SAME: norecurse
; CHECK-NOT: readonly
; CHECK-NEXT: void @foo()
define void @foo() {
  %tmp = load volatile i32, ptr @g
  ret void
}
