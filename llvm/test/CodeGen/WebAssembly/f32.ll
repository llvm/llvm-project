; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 32-bit floating-point operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare float @llvm.fabs.f32(float)
declare float @llvm.copysign.f32(float, float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.ceil.f32(float)
declare float @llvm.floor.f32(float)
declare float @llvm.trunc.f32(float)
declare float @llvm.nearbyint.f32(float)
declare float @llvm.rint.f32(float)

; CHECK-LABEL: fadd32:
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .param f32{{$}}
; CHECK-NEXT: .result f32{{$}}
; CHECK-NEXT: .local f32, f32, f32{{$}}
; CHECK-NEXT: get_local push, 1{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
; CHECK-NEXT: f32.add push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
; CHECK-NEXT: return (get_local 4){{$}}
define float @fadd32(float %x, float %y) {
  %a = fadd float %x, %y
  ret float %a
}

; CHECK-LABEL: fsub32:
; CHECK: f32.sub push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define float @fsub32(float %x, float %y) {
  %a = fsub float %x, %y
  ret float %a
}

; CHECK-LABEL: fmul32:
; CHECK: f32.mul push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define float @fmul32(float %x, float %y) {
  %a = fmul float %x, %y
  ret float %a
}

; CHECK-LABEL: fdiv32:
; CHECK: f32.div push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define float @fdiv32(float %x, float %y) {
  %a = fdiv float %x, %y
  ret float %a
}

; CHECK-LABEL: fabs32:
; CHECK: f32.abs push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define float @fabs32(float %x) {
  %a = call float @llvm.fabs.f32(float %x)
  ret float %a
}

; CHECK-LABEL: fneg32:
; CHECK: f32.neg push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define float @fneg32(float %x) {
  %a = fsub float -0., %x
  ret float %a
}

; CHECK-LABEL: copysign32:
; CHECK: f32.copysign push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define float @copysign32(float %x, float %y) {
  %a = call float @llvm.copysign.f32(float %x, float %y)
  ret float %a
}

; CHECK-LABEL: sqrt32:
; CHECK: f32.sqrt push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define float @sqrt32(float %x) {
  %a = call float @llvm.sqrt.f32(float %x)
  ret float %a
}

; CHECK-LABEL: ceil32:
; CHECK: f32.ceil push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define float @ceil32(float %x) {
  %a = call float @llvm.ceil.f32(float %x)
  ret float %a
}

; CHECK-LABEL: floor32:
; CHECK: f32.floor push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define float @floor32(float %x) {
  %a = call float @llvm.floor.f32(float %x)
  ret float %a
}

; CHECK-LABEL: trunc32:
; CHECK: f32.trunc push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define float @trunc32(float %x) {
  %a = call float @llvm.trunc.f32(float %x)
  ret float %a
}

; CHECK-LABEL: nearest32:
; CHECK: f32.nearest push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define float @nearest32(float %x) {
  %a = call float @llvm.nearbyint.f32(float %x)
  ret float %a
}

; CHECK-LABEL: nearest32_via_rint:
; CHECK: f32.nearest push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define float @nearest32_via_rint(float %x) {
  %a = call float @llvm.rint.f32(float %x)
  ret float %a
}
