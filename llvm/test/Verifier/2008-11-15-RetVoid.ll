; RUN: not llvm-as < %s |& grep {returns non-void in Function of void return}

define void @foo() {
  ret i32 0
}
