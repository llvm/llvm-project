; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: invalid basic block in uselistorder_bb
define void @foo() {
  unreachable
}
uselistorder_bb @foo, %bb, { 1, 0 }
