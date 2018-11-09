; RUN: llc < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16-a0:16:16"
target triple = "msp430---elf"

define void @test() #0 {
; CHECK-LABEL: test:
; CHECK: sub.w #2, r1
  %1 = alloca i8, align 1
; CHECK-NEXT: mov.b #0, 1(r1)
  store i8 0, i8* %1, align 1
; CHECK-NEXT: add.w #2, r1
; CHECK-NEXT: ret
  ret void
}

attributes #0 = { nounwind "no-frame-pointer-elim"="false" }
