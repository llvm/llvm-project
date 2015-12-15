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
declare float @llvm.fma.f32(float, float, float)

; CHECK-LABEL: fadd32:
; CHECK-NEXT: .param f32, f32{{$}}
; CHECK-NEXT: .result f32{{$}}
; CHECK-NEXT: f32.add $push0=, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @fadd32(float %x, float %y) {
  %a = fadd float %x, %y
  ret float %a
}

; CHECK-LABEL: fsub32:
; CHECK: f32.sub $push0=, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @fsub32(float %x, float %y) {
  %a = fsub float %x, %y
  ret float %a
}

; CHECK-LABEL: fmul32:
; CHECK: f32.mul $push0=, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @fmul32(float %x, float %y) {
  %a = fmul float %x, %y
  ret float %a
}

; CHECK-LABEL: fdiv32:
; CHECK: f32.div $push0=, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @fdiv32(float %x, float %y) {
  %a = fdiv float %x, %y
  ret float %a
}

; CHECK-LABEL: fabs32:
; CHECK: f32.abs $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @fabs32(float %x) {
  %a = call float @llvm.fabs.f32(float %x)
  ret float %a
}

; CHECK-LABEL: fneg32:
; CHECK: f32.neg $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @fneg32(float %x) {
  %a = fsub float -0., %x
  ret float %a
}

; CHECK-LABEL: copysign32:
; CHECK: f32.copysign $push0=, $0, $1{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @copysign32(float %x, float %y) {
  %a = call float @llvm.copysign.f32(float %x, float %y)
  ret float %a
}

; CHECK-LABEL: sqrt32:
; CHECK: f32.sqrt $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @sqrt32(float %x) {
  %a = call float @llvm.sqrt.f32(float %x)
  ret float %a
}

; CHECK-LABEL: ceil32:
; CHECK: f32.ceil $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @ceil32(float %x) {
  %a = call float @llvm.ceil.f32(float %x)
  ret float %a
}

; CHECK-LABEL: floor32:
; CHECK: f32.floor $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @floor32(float %x) {
  %a = call float @llvm.floor.f32(float %x)
  ret float %a
}

; CHECK-LABEL: trunc32:
; CHECK: f32.trunc $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @trunc32(float %x) {
  %a = call float @llvm.trunc.f32(float %x)
  ret float %a
}

; CHECK-LABEL: nearest32:
; CHECK: f32.nearest $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @nearest32(float %x) {
  %a = call float @llvm.nearbyint.f32(float %x)
  ret float %a
}

; CHECK-LABEL: nearest32_via_rint:
; CHECK: f32.nearest $push0=, $0{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @nearest32_via_rint(float %x) {
  %a = call float @llvm.rint.f32(float %x)
  ret float %a
}

; Min and max tests. LLVM currently only forms fminnan and fmaxnan nodes in
; cases where there's a single fcmp with a select and it can prove that one
; of the arms is never NaN, so we only test that case. In the future if LLVM
; learns to form fminnan/fmaxnan in more cases, we can write more general
; tests.

; CHECK-LABEL: fmin32:
; CHECK: f32.min $push1=, $0, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define float @fmin32(float %x) {
  %a = fcmp ult float %x, 0.0
  %b = select i1 %a, float %x, float 0.0
  ret float %b
}

; CHECK-LABEL: fmax32:
; CHECK: f32.max $push1=, $0, $pop0{{$}}
; CHECK-NEXT: return $pop1{{$}}
define float @fmax32(float %x) {
  %a = fcmp ugt float %x, 0.0
  %b = select i1 %a, float %x, float 0.0
  ret float %b
}

; CHECK-LABEL: fma32:
; CHECK: {{^}} f32.call $push0=, fmaf, $0, $1, $2{{$}}
; CHECK-NEXT: return $pop0{{$}}
define float @fma32(float %a, float %b, float %c) {
  %d = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %d
}
