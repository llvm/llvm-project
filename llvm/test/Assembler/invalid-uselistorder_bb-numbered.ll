; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: invalid numeric label in uselistorder_bb

@ba1 = constant ptr blockaddress (@foo, %1)

define void @foo() {
  br label %1
  unreachable
}

uselistorder_bb @foo, %1, { 1, 0 }
