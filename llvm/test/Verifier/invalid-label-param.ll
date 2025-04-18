; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define void @invalid_arg_type(label %p) {
; CHECK: Function argument cannot be of label type!
  ret void
}

