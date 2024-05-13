; Test to make sure malloc's bitcast does not block detection of a store 
; to aliased memory; GVN should not optimize away the load in this program.
; RUN: opt < %s -passes=gvn -S | FileCheck %s

define i64 @test() {
  %1 = tail call ptr @malloc(i64 mul (i64 4, i64 ptrtoint (ptr getelementptr (i64, ptr null, i64 1) to i64))) ; <ptr> [#uses=2]
  store i8 42, ptr %1
  %Y = load i64, ptr %1                               ; <i64> [#uses=1]
  ret i64 %Y
; CHECK: %Y = load i64, ptr %1
; CHECK: ret i64 %Y
}

declare noalias ptr @malloc(i64)
