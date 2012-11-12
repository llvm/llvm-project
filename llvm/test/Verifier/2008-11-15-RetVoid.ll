; RUN: not llvm-as < %s 2>&1 | grep "value doesn't match function result type 'void'"

define void @foo() {
  ret i32 0
}
