; RUN: opt < %s -aa-pipeline=basic-aa -passes=dse -S | FileCheck %s

define void @test(ptr %P) {
; CHECK: store i32 0, ptr %X
  %Q = getelementptr {i32,i32}, ptr %P, i32 1
  %X = getelementptr {i32,i32}, ptr %Q, i32 0, i32 1
  %Y = getelementptr {i32,i32}, ptr %Q, i32 1, i32 1
  store i32 0, ptr %X
  store i32 1, ptr %Y
  ret void
}
