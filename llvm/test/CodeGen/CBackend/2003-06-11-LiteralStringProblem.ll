; RUN: llvm-as < %s | llc -march=c

@version = global [3 x i8] c"1\00\00"
