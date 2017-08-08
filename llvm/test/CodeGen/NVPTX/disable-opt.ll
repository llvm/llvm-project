; RUN: llc < %s -march=nvptx -mcpu=sm_20 -O0 | FileCheck %s

define void @foo(i32* %output) {
; CHECK-LABEL: .visible .func foo(
entry:
  %local = alloca i32
; CHECK: __local_depot
  store i32 1, i32* %local
  %0 = load i32, i32* %local
  store i32 %0, i32* %output
  ret void
}
