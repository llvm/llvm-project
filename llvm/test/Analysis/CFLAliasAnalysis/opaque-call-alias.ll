; We previously had a case where we would put results from a no-args call in
; its own stratified set. This would make cases like the one in @test say that
; nothing (except %Escapes and %Arg) can alias

; RUN: opt < %s -disable-basicaa -cfl-aa -aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s

; CHECK:     Function: test
; CHECK:     NoAlias: i8* %Arg, i8* %Escapes
; CHECK:     MayAlias: i8* %Arg, i8* %Retrieved
; CHECK:     MayAlias: i8* %Escapes, i8* %Retrieved
define void @test(i8* %Arg) {
  %Noalias = alloca i8
  %Escapes = alloca i8
  call void @set_thepointer(i8* %Escapes)
  %Retrieved = call i8* @get_thepointer()
  ret void
}

declare void @set_thepointer(i8* %P)
declare i8* @get_thepointer()
