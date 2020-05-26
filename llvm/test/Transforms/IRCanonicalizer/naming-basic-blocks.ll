; RUN: opt -S --ir-canonicalizer --rename-all < %s | FileCheck %s

define i32 @foo(i32 %a0) {
; CHECK: bb{{([0-9]{5})}}
entry:
  %a = add i32 %a0, 2
  ret i32 %a
}