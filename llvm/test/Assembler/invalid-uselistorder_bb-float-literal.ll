; RUN: not llvm-as < %s -disable-output 2>&1 | FileCheck %s
; CHECK: error: unexpected floating-point literal

@ba1 = constant ptr blockaddress (@foo, %1)

define void @foo() {
  br label %1
  unreachable
}

uselistorder_bb 1.0, %1, { 1, 0 }
