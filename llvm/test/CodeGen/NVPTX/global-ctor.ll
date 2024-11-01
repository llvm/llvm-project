; RUN: not --crash llc < %s -march=nvptx -mcpu=sm_20 2>&1 | FileCheck %s

; Check that llc dies when given a nonempty global ctor.
@llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @foo, ptr null }]

; CHECK: ERROR: Module has a nontrivial global ctor
define internal void @foo() {
  ret void
}
