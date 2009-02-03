; RUN: llvm-as %s -o /dev/null -f

@X = external global i32*
@X1 = external global %T* 
@X2 = external global i32*

%T = type i32
