; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 -O0 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 -O0 | %ptxas-verify %}

define void @foo(ptr %output) {
; CHECK-LABEL: .visible .func foo(
entry:
  %local = alloca i32
; CHECK: __local_depot
  store i32 1, ptr %local
  %0 = load i32, ptr %local
  store i32 %0, ptr %output
  ret void
}
