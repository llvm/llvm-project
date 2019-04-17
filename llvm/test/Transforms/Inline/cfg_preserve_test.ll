; This test ensures that inlining an "empty" function does not destroy the CFG
;
; RUN: opt < %s -inline -S | FileCheck %s

define i32 @func(i32 %i) {
  ret i32 %i
}


define i32 @main() {
; CHECK-LABEL: define i32 @main()
entry:
  %X = call i32 @func(i32 7)
; CHECK-NOT: call
; CHECK-NOT: br

  ret i32 %X
; CHECK: ret i32 7
}

