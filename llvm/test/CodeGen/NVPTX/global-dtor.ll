; RUN: not llc < %s -march=nvptx -mcpu=sm_20 2>&1 | FileCheck %s

; Check that llc dies when given a nonempty global dtor.
@llvm.global_dtors = appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 65535, void ()* @foo, i8* null }]

; CHECK: ERROR: Module has a nontrivial global dtor
define internal void @foo() {
  ret void
}
