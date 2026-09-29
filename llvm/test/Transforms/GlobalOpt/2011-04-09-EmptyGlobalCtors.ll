; RUN: opt < %s -passes=globalopt -disable-output

%0 = type { i32, ptr, ptr }
@llvm.global_ctors = appending global [0 x %0] zeroinitializer

