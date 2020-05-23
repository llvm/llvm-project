; RUN: opt -S --ir-canonicalizer < %s | FileCheck %s

; CHECK: @foo(i32 %a0, i32 %a1)
define i32 @foo(i32, i32) {
  %tmp = mul i32 %0, %1
  ret i32 %tmp
}