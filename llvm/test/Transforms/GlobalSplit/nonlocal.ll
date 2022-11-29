; RUN: opt -S -passes=globalsplit %s | FileCheck %s

target datalayout = "e-p:64:64"
target triple = "x86_64-unknown-linux-gnu"

; CHECK: @global =
@global = constant { [2 x i8* ()*], [1 x i8* ()*] } {
  [2 x i8* ()*] [i8* ()* @f, i8* ()* @g],
  [1 x i8* ()*] [i8* ()* @h]
}

define i8* @f() {
  ret i8* bitcast (i8* ()** getelementptr ({ [2 x i8* ()*], [1 x i8* ()*] }, { [2 x i8* ()*], [1 x i8* ()*] }* @global, i32 0, inrange i32 0, i32 0) to i8*)
}

define i8* @g() {
  ret i8* null
}

define i8* @h() {
  ret i8* null
}

define void @foo() {
  %p = call i1 @llvm.type.test(i8* null, metadata !"")
  ret void
}

declare i1 @llvm.type.test(i8*, metadata) nounwind readnone
