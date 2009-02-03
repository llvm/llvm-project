; RUN: llvm-as < %s | llc -march=c

@bob = external global i32              ; <i32*> [#uses=0]

