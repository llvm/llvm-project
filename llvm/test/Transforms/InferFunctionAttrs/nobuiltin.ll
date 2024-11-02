; RUN: opt -S -passes=inferattrs < %s | FileCheck %s

; CHECK: Function Attrs: nobuiltin allocsize(0){{$}}
; CHECK: declare i8* @_Znwm(i32)
declare i8* @_Znwm(i32) nobuiltin allocsize(0)
