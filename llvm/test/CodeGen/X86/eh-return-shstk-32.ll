; RUN: llc -mtriple=i386-pc-linux -mattr=+shstk -verify-machineinstrs < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64"
target triple = "i386-pc-linux"

define void @test(i32 %offset, ptr %handler) {
entry:
  call void @llvm.eh.return.i32(i32 %offset, ptr %handler)
  unreachable
}

; CHECK-LABEL: test:
; CHECK:       movl %ecx, %esp
; CHECK-NEXT:  popl %ecx
; CHECK-NEXT:  jmpl *%ecx

declare void @llvm.eh.return.i32(i32, ptr)