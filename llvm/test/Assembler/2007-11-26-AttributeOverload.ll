; RUN: not llvm-as < %s > /dev/null 2>&1

declare i32 @atoi(ptr) nounwind readonly
declare i32 @atoi(ptr)
