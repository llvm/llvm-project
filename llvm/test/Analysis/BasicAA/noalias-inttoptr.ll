; RUN: opt < %s -basic-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
target datalayout = "p:64:64"

declare void @escape(i8* %ptr)

; Verify that unescaped noalias parameter does not alias inttoptr
define void @test1(i8* noalias %P, i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test1:
  ; CHECK: NoAlias:	i8* %P, i8* %Q
  %Q = inttoptr i64 %Q_as_int to i8*
  store i8 0, i8* %P
  store i8 1, i8* %Q
  ret void
}

; Verify that unescaped alloca does not alias inttoptr
define void @test2(i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test2:
  ; CHECK: NoAlias:	i8* %P, i8* %Q
  %P = alloca i8
  %Q = inttoptr i64 %Q_as_int to i8*
  store i8 0, i8* %P
  store i8 1, i8* %Q
  ret void
}

; Verify that escaped noalias parameter may alias inttoptr
define void @test3(i8* noalias %P, i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test3:
  ; CHECK: MayAlias:	i8* %P, i8* %Q
  call void @escape(i8* %P)
  %Q = inttoptr i64 %Q_as_int to i8*
  store i8 0, i8* %P
  store i8 1, i8* %Q
  ret void
}

; Verify that escaped alloca may alias inttoptr
define void @test4(i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test4:
  ; CHECK: MayAlias:	i8* %P, i8* %Q
  %P = alloca i8
  call void @escape(i8* %P)
  %Q = inttoptr i64 %Q_as_int to i8*
  store i8 0, i8* %P
  store i8 1, i8* %Q
  ret void
}


; Verify that global may alias inttoptr
@G = external global i8
define void @test5(i64 %Q_as_int) {
  ; CHECK-LABEL: Function: test5:
  ; CHECK: MayAlias:	i8* %Q, i8* @G
  %Q = inttoptr i64 %Q_as_int to i8*
  store i8 0, i8* @G
  store i8 1, i8* %Q
  ret void
}
