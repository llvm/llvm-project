; RUN: llvm-as < %s | llvm-dis
; RUN: verify-uselistorder %s
; PR3372

@X = global ptr @0
@0 = global i32 4

