; Weak variables should be preserved by global DCE!

; RUN: llvm-as < %s | opt -globaldce | llvm-dis | grep @A


@A = weak global i32 54
