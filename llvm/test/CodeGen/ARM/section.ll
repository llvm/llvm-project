; RUN: llc < %s -mtriple=arm-linux | FileCheck %s

; CHECK: .section .dtors,"aw",%progbits
; CHECK: __DTOR_END__:
@__DTOR_END__ = internal global [1 x i32] zeroinitializer, section ".dtors"       ; <[1 x i32]*> [#uses=0]

