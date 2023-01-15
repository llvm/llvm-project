; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn -S | FileCheck %s

declare noalias ptr @noalias()

define i32 @test(i32 %x) {
; CHECK: load i32, ptr %a
  %a = call ptr @noalias()
  store i32 1, ptr %a
  %b = getelementptr i32, ptr %a, i32 %x
  store i32 2, ptr %b

  %c = load i32, ptr %a
  ret i32 %c
}
