; RUN: not llc -mtriple=aarch64-none-linux-gnu -mattr=+sve -o - %s 2>&1 | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

; CHECK: error: couldn't allocate input reg for constraint 'Upa'
; CHECK: error: couldn't allocate input reg for constraint 'r'
; CHECK: error: couldn't allocate output register for constraint 'w'
; CHECK: error: unknown token in expression

define <vscale x 16 x i1> @foo1(ptr %in) {
entry:
  %0 = load i32, ptr %in, align 4
  %1 = call <vscale x 16 x i1> asm sideeffect "mov $0.b, $1.b \0A", "=@3Upa,@3Upa"(i32 %0)
  ret <vscale x 16 x i1> %1
}

define <vscale x 4 x float> @foo2(ptr %in) {
entry:
  %0 = load <vscale x 4 x i32>, ptr %in, align 16
  %1 = call <vscale x 4 x float> asm sideeffect "ptrue p0.s, #1 \0Afabs $0.s, p0/m, $1.s \0A", "=w,r"(<vscale x 4 x i32> %0)
  ret <vscale x 4 x float> %1
}

define <vscale x 16 x i1> @foo3(ptr %in) {
entry:
  %0 = load <vscale x 16 x i1>, ptr %in, align 2
  %1 = call <vscale x 16 x i1> asm sideeffect "mov $0.b, $1.b \0A", "=&w,w"(<vscale x 16 x i1> %0)
  ret <vscale x 16 x i1> %1
}

define half @foo4(ptr %inp, ptr %inv) {
entry:
  %0 = load <vscale x 16 x i1>, ptr %inp, align 2
  %1 = load <vscale x 8 x half>, ptr %inv, align 16
  %2 = call half asm "fminv ${0:h}, $1, $2.h", "=r,@3Upl,w"(<vscale x 16 x i1> %0, <vscale x 8 x half> %1)
  ret half %2
}
