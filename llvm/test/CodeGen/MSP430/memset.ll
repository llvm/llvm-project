; RUN: llc < %s | FileCheck %s
target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

@buf = external global i8*

; Function Attrs: nounwind
define void @test() nounwind {
entry:
; CHECK-LABEL: test:
  %0 = load i8*, i8** @buf, align 2
; CHECK: mov.w &buf, r15
; CHECK-NEXT: mov.w #5, r14
; CHECK-NEXT: mov.w #128, r13
; CHECK-NEXT: call #memset
  call void @llvm.memset.p0i8.i16(i8* %0, i8 5, i16 128, i32 1, i1 false)
  ret void
}

; Function Attrs: nounwind
declare void @llvm.memset.p0i8.i16(i8* nocapture, i8, i16, i32, i1) nounwind

