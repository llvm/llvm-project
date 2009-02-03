; RUN: llvm-as < %s | llc -march=c

@MyIntList = external global { \2*, i32 }
