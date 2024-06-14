; RUN: llc < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i386-pc-windows-msvc"

define x86_fastcallcc i32 @test1(i32 inreg %V, ptr byval([65533 x i8]) %p_arg) {
  ret i32 %V
}
; CHECK-LABEL: @test1@65540:
; CHECK:      movl %ecx, %eax
; CHECK-NEXT: popl %ecx
; CHECK-NEXT: addl $65536, %esp
; CHECK-NEXT: pushl %ecx
; CHECK-NEXT: retl

define x86_stdcallcc void @test2(ptr byval([65533 x i8]) %p_arg) {
  ret void
}
; CHECK-LABEL: _test2@65536:
; CHECK:      popl %ecx
; CHECK-NEXT: addl $65536, %esp
; CHECK-NEXT: pushl %ecx
; CHECK-NEXT: retl
