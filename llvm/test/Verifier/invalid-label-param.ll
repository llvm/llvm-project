; RUN: not llvm-as < %s 2>&1 | FileCheck %s

define void @invalid_arg_type(i32 %0) {
1:
  call void @foo(label %1)
  ret void
}

declare void @foo(label)
; CHECK: invalid type for function argument
