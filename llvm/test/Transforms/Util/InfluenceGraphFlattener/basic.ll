; RUN: opt -disable-output -passes='print<influence-graph-flattener>' %s 2>&1 | FileCheck %s

; CHECK: %a = add i32 2, 3 is used by:
; CHECK-NEXT: ret i32 %a
define i32 @foo() {
  %a = add i32 2, 3
  ret i32 %a
}
