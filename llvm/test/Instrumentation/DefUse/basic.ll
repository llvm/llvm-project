; RUN: opt -passes=def-use-instrumentation -S %s | FileCheck %s

define i32 @memory_test(ptr %p, i32 %x) {
entry:
  store i32 %x, ptr %p
  %value = load i32, ptr %p
  ret i32 %value
}
