; RUN: opt < %s -basic-aa -aa-eval -disable-output 2>&1 | FileCheck %s

declare noalias i32* @_Znwj(i32 %x) nounwind

; CHECK: 1 no alias response

define i32 @foo() {
  %A = call i32* @_Znwj(i32 4)
  %B = call i32* @_Znwj(i32 4)
  store i32 1, i32* %A
  store i32 2, i32* %B
  %C = load i32, i32* %A
  ret i32 %C
}
