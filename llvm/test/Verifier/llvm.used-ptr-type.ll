; RUN: llvm-as < %s -disable-output

@a = global i32 42
@llvm.used = appending global [1 x ptr] [ptr @a], section "llvm.metadata"
