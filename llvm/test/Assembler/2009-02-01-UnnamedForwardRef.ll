; RUN: llvm-as < %s | llvm-dis
; PR3372

@X = global i32* @0
global i32 4

