; We previously had a case where we would put results from a no-args call in
; its own stratified set. This would make cases like the one in @test say that
; nothing (except %Escapes and %Arg) can alias

; RUN: opt < %s -aa-pipeline=cfl-steens-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK:     Function: test
; CHECK:     NoAlias: i8* %Arg, i8* %Escapes
; CHECK:     MayAlias: i8* %Arg, i8* %Retrieved
; CHECK:     MayAlias: i8* %Escapes, i8* %Retrieved
define void @test(i8* %Arg) {
  %Noalias = alloca i8
  %Escapes = alloca i8
  load i8, i8* %Arg
  load i8, i8* %Escapes
  call void @set_thepointer(i8* %Escapes)
  %Retrieved = call i8* @get_thepointer()
  load i8, i8* %Retrieved
  ret void
}

declare void @set_thepointer(i8* %P)
declare i8* @get_thepointer()
