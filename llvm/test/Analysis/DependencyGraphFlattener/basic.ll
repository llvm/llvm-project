; RUN: opt -disable-output -passes='print<dependency-graph-flattener>' %s 2>&1 | FileCheck %s

; CHECK: ret i32 %a uses:
; CHECK-NEXT: %a = add i32 2, 3
define i32 @foo() {
  %a = add i32 2, 3
  ret i32 %a
}
