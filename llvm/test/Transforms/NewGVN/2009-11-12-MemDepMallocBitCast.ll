; Test to make sure malloc's bitcast does not block detection of a store 
; to aliased memory; GVN should not optimize away the load in this program.
; RUN: opt < %s -passes=newgvn -S | FileCheck %s

define i64 @test() {
  %1 = tail call i8* @malloc(i64 mul (i64 4, i64 ptrtoint (i64* getelementptr (i64, i64* null, i64 1) to i64))) ; <i8*> [#uses=2]
  store i8 42, i8* %1
  %X = bitcast i8* %1 to i64*                     ; <i64*> [#uses=1]
  %Y = load i64, i64* %X                               ; <i64> [#uses=1]
  ret i64 %Y
; CHECK: %Y = load i64, i64* %X
; CHECK: ret i64 %Y
}

declare noalias i8* @malloc(i64)
