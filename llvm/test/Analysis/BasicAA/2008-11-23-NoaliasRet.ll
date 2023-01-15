; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -disable-output 2>&1 | FileCheck %s

declare noalias ptr @_Znwj(i32 %x) nounwind

; CHECK: 1 no alias response

define i32 @foo() {
  %A = call ptr @_Znwj(i32 4)
  %B = call ptr @_Znwj(i32 4)
  store i32 1, ptr %A
  store i32 2, ptr %B
  %C = load i32, ptr %A
  ret i32 %C
}
