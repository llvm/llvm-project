; Before MIPS32, single-precision arithmetic can't use the odd FPRs when the
; FPRs are 32 bits wide.

; RUN: llc -mtriple=mipsel -mcpu=mips1 < %s | FileCheck %s --check-prefix=EVEN
; RUN: llc -mtriple=mipsel -mcpu=mips2 < %s | FileCheck %s --check-prefix=EVEN
; RUN: llc -mtriple=mipsel -mcpu=mips3 -mattr=-fp64 < %s \
; RUN:   | FileCheck %s --check-prefix=EVEN
; RUN: llc -mtriple=mipsel -mcpu=mips4 -mattr=-fp64 < %s \
; RUN:   | FileCheck %s --check-prefix=EVEN

; RUN: llc -mtriple=mipsel -mcpu=mips32 < %s | FileCheck %s --check-prefix=ODD
; RUN: llc -mtriple=mipsel -mcpu=mips3 < %s | FileCheck %s --check-prefix=ODD
; RUN: llc -mtriple=mipsel -mcpu=mips1 -mattr=+single-float < %s \
; RUN:   | FileCheck %s --check-prefix=ODD
; RUN: llc -mtriple=mipsel -mcpu=r5900 < %s | FileCheck %s --check-prefix=R5900

; RUN: llc -mtriple=mipsel -mcpu=mips1 -filetype=obj < %s \
; RUN:   | llvm-readobj -A - | FileCheck %s --check-prefix=FLAGS-EVEN
; RUN: llc -mtriple=mipsel -mcpu=mips32 -filetype=obj < %s \
; RUN:   | llvm-readobj -A - | FileCheck %s --check-prefix=FLAGS-ODD

; EVEN:     .module nooddspreg
; ODD-NOT:  .module nooddspreg
; R5900-NOT: .module nooddspreg

; EVEN-LABEL: many:
; EVEN-NOT:   {{\$f(1|3|5|7|9|11|13|15|17|19|21|23|25|27|29|31)([^0-9]|$)}}
; EVEN:       .end many
; ODD-LABEL:  many:
; ODD:        {{(add|mul)\.s.*\$f(1|3|5|7|9|11|13|15|17|19|21|23|25|27|29|31)([^0-9]|$)}}
; ODD:        .end many

define float @many(ptr %p) {
  %a0 = load float, ptr %p
  %q1 = getelementptr float, ptr %p, i32 1
  %a1 = load float, ptr %q1
  %q2 = getelementptr float, ptr %p, i32 2
  %a2 = load float, ptr %q2
  %q3 = getelementptr float, ptr %p, i32 3
  %a3 = load float, ptr %q3
  %q4 = getelementptr float, ptr %p, i32 4
  %a4 = load float, ptr %q4
  %q5 = getelementptr float, ptr %p, i32 5
  %a5 = load float, ptr %q5
  %m0 = fmul float %a0, %a1
  %m1 = fmul float %a2, %a3
  %m2 = fmul float %a4, %a5
  %m3 = fmul float %a0, %a5
  %s0 = fadd float %m0, %m1
  %s1 = fadd float %m2, %m3
  %s2 = fadd float %s0, %s1
  ret float %s2
}

; The fastcc f32 arguments are also passed in the even FPRs.

; EVEN-LABEL: callee:
; EVEN:       add.s $f0, $f0, $f2
; ODD-LABEL:  callee:
; ODD:        add.s $f0, $f0, $f1
; R5900-LABEL: callee:
; R5900:       mov.s $f14, $f1

define internal fastcc float @callee(float %a, float %b) {
  %s = fadd float %a, %b
  ret float %s
}

define float @caller(float %a, float %b) {
  %r = call fastcc float @callee(float %a, float %b)
  ret float %r
}

; FLAGS-EVEN:      Flags 1 [ (0x0)
; FLAGS-EVEN-NEXT: ]
; FLAGS-ODD:       Flags 1 [ (0x1)
; FLAGS-ODD-NEXT:    ODDSPREG (0x1)
