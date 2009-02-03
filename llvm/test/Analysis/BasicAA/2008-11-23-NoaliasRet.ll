; RUN: llvm-as < %s | opt -aa-eval |& grep {1 no alias response}

declare noalias i32* @_Znwj(i32 %x) nounwind

define i32 @foo() {
  %A = call i32* @_Znwj(i32 4)
  %B = call i32* @_Znwj(i32 4)
  store i32 1, i32* %A
  store i32 2, i32* %B
  %C = load i32* %A
  ret i32 %C
}
