; RUN: not llvm-as %s |& grep {found end of file when expecting more instructions}

define void @foo() {
