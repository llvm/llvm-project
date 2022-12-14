; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s

@X = external global ptr
@X1 = external global ptr 
@X2 = external global ptr

%T = type {i32}
