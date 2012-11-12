; RUN: llc -O0 -disable-fp-elim < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16"
target triple = "msp430---elf"

define void @fp() nounwind {
entry:
; CHECK: fp:
; CHECK: push.w r4
; CHECK: mov.w r1, r4
; CHECK: sub.w #2, r1
  %i = alloca i16, align 2
; CHECK: mov.w #0, -2(r4)
  store i16 0, i16* %i, align 2
; CHECK: pop.w r4
  ret void
}
