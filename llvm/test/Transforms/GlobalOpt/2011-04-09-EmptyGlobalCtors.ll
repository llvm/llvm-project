; RUN: opt < %s -globalopt -disable-output

%0 = type { i32, void ()* }
@llvm.global_ctors = appending global [0 x %0] zeroinitializer

