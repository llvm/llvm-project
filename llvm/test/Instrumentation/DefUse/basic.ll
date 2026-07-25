; RUN: opt -passes=def-use-instrumentation -S %s | FileCheck %s

declare void @nobodyfunc ()

define i32 @foo(i32 %x) {
entry:
  %a = add i32 %x, 1
  %b = mul i32 %a, 2
  ret i32 %b
}