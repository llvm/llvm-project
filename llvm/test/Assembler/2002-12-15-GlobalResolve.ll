; RUN: llvm-as %s -disable-output
; RUN: verify-uselistorder %s

@X = external global ptr
@X1 = external global ptr 
@X2 = external global ptr

%T = type {i32}
