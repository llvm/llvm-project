; RUN: opt -S --ir-canonicalizer --rename-all < %s | FileCheck %s

define i32 @foo(i32 %a0) {
entry:
; CHECK: %"vl{{([0-9]{5})}}(%a0, 2)"
  %a = add i32 %a0, 2
; CHECK: %"op{{([0-9]{5})}}(vl{{([0-9]{5})}})"
  %b = add i32 %a, 6
; CHECK: %"op{{([0-9]{5})}}(8, op{{([0-9]{5})}}(6, vl{{([0-9]{5})}}(%a0, 2)))"
  %c = add i32 %b, 8
  ret i32 %c
}