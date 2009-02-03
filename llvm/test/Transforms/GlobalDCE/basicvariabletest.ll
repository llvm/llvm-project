; RUN: llvm-as < %s | opt -globaldce | llvm-dis | not grep global

@X = external global i32
@Y = internal global i32 7

