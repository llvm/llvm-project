; RUN: llvm-as < %s

define noalias i8* @_Znwj(i32 %x) nounwind {
  %A = malloc i8, i32 %x
  ret i8* %A
}
