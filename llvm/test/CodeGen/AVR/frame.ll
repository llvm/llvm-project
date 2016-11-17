; RUN: llc -mattr=mul < %s -march=avr | FileCheck %s

declare float @dsin(float)
declare float @dcos(float)
declare float @dasin(float)

; Test prologue and epilogue insertion
define float @f3(float %days) {
entry:
; CHECK-LABEL: f3:
; prologue code:
; CHECK: push r28
; CHECK: push r29
; CHECK: in r28, 61
; CHECK-NEXT: in r29, 62
; CHECK-NEXT: sbiw r28, [[SIZE:[0-9]+]]
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
; epilogue code:
; CHECK: adiw r28, [[SIZE]]
; CHECK-NEXT: in r0, 63
; CHECK-NEXT: cli
; CHECK-NEXT: out 62, r29
; CHECK-NEXT: out 63, r0
; CHECK-NEXT: out 61, r28
; CHECK: pop r29
; CHECK: pop r28
  %mul = fmul float %days, 0x3FEF8A6C60000000
  %add = fadd float %mul, 0x40718776A0000000
  %mul1 = fmul float %days, 0x3FEF8A09A0000000
  %add2 = fadd float %mul1, 0x4076587740000000
  %mul3 = fmul float %days, 0x3E81B35CC0000000
  %sub = fsub float 0x3FFEA235C0000000, %mul3
  %call = call float @dsin(float %add2)
  %mul4 = fmul float %sub, %call
  %mul5 = fmul float %days, 0x3E27C04CA0000000
  %sub6 = fsub float 0x3F94790B80000000, %mul5
  %mul7 = fmul float %add2, 2.000000e+00
  %call8 = call float @dsin(float %mul7)
  %mul9 = fmul float %sub6, %call8
  %add10 = fadd float %mul4, %mul9
  %add11 = fadd float %add, %add10
  %mul12 = fmul float %days, 0x3E13C5B640000000
  %sub13 = fsub float 0x3F911C1180000000, %mul12
  %mul14 = fmul float %add, 2.000000e+00
  %call15 = call float @dsin(float %mul14)
  %mul16 = fmul float %call15, 0x3FF1F736C0000000
  %mul17 = fmul float %sub13, 2.000000e+00
  %mul19 = fmul float %mul17, %call
  %sub20 = fsub float %mul16, %mul19
  %mul21 = fmul float %sub13, 4.000000e+00
  %mul22 = fmul float %mul21, 0x3FF1F736C0000000
  %mul24 = fmul float %mul22, %call
  %call26 = call float @dcos(float %mul14)
  %mul27 = fmul float %mul24, %call26
  %add28 = fadd float %sub20, %mul27
  %call29 = call float @dsin(float %add11)
  %mul30 = fmul float %call29, 0x3FF0AB6960000000
  %call31 = call float @dasin(float %mul30)
  %add32 = fadd float %call31, %add28
  ret float %add32
}
