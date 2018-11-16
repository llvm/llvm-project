; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16-a0:16:16"
target triple = "msp430---elf"

declare i32 @direct(i32 %a)

define i32 @test_direct(i32 %a) nounwind {
; CHECK-LABEL: test_direct:
; CHECK: call #direct
  %1 = call i32 @direct(i32 %a)
  ret i32 %1
}

define i16 @test_indirect(i16 (i16)* %a, i16 %b) nounwind {
; CHECK-LABEL: test_indirect:
; CHECK: mov	r12, r14
; CHECK: mov	r13, r12
; CHECK: call	r14
  %1 = call i16 %a(i16 %b)
  ret i16 %1
}
