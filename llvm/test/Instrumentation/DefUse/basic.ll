; RUN: opt -passes=def-use-instrumentation -S %s | FileCheck %s

define i32 @main() {
entry:
  %x = alloca i32
  ret i32 0
}

define i32 @foo() {
entry:
  %x = alloca i32
  ret i32 0
}

define i32 @bar() {
entry:
  %x = alloca i32
  ret i32 0
}
