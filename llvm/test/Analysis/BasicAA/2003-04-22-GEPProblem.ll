; RUN: opt < %s -aa-pipeline=basic-aa -passes=gvn,instcombine -S | FileCheck %s

; BasicAA was incorrectly concluding that P1 and P2 didn't conflict!

define i32 @test(ptr %Ptr, i64 %V) {
; CHECK: sub i32 %X, %Y
  %P2 = getelementptr i32, ptr %Ptr, i64 1
  %P1 = getelementptr i32, ptr %Ptr, i64 %V
  %X = load i32, ptr %P1
  store i32 5, ptr %P2
  %Y = load i32, ptr %P1
  %Z = sub i32 %X, %Y
  ret i32 %Z
}
