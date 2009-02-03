; RUN: not llvm-as %s -o /dev/null -f |& grep {redefinition of global '@B'}

@B = global i32 7
@B = global i32 7
