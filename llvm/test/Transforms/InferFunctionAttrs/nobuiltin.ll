; RUN: opt -S -passes=inferattrs < %s | FileCheck %s

; CHECK: Function Attrs: nobuiltin allocsize(0){{$}}
; CHECK: declare ptr @_Znwm(i32)
declare ptr @_Znwm(i32) nobuiltin allocsize(0)
