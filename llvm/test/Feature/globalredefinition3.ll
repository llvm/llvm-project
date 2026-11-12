; RUN: not llvm-as %s -disable-output 2>&1 | grep "redefinition of global '@B'"

@B = global i32 7
@B = global i32 7
