; RUN: llvm-as < %s | llc -march=c | grep common | grep X

@X = linkonce global i32 5
