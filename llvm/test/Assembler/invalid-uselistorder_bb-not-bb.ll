; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: expected basic block in uselistorder_bb
define i32 @foo(i32 %arg) {
  ret i32 %arg
}
uselistorder_bb @foo, %arg, { 1, 0 }
