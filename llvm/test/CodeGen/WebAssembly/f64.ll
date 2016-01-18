; RUN: llc < %s -asm-verbose=false | FileCheck %s

; Test that basic 64-bit floating-point operations assemble as expected.

target datalayout = "e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare double @llvm.fabs.f64(double)
declare double @llvm.copysign.f64(double, double)
declare double @llvm.sqrt.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.trunc.f64(double)
declare double @llvm.nearbyint.f64(double)
declare double @llvm.rint.f64(double)

; CHECK-LABEL: fadd64:
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .param f64{{$}}
; CHECK-NEXT: .result f64{{$}}
; CHECK-NEXT: .local f64, f64, f64{{$}}
; CHECK-NEXT: get_local 1{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: get_local 0{{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
; CHECK-NEXT: add (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
; CHECK-NEXT: return (get_local 4){{$}}
define double @fadd64(double %x, double %y) {
  %a = fadd double %x, %y
  ret double %a
}

; CHECK-LABEL: fsub64:
; CHECK: sub (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define double @fsub64(double %x, double %y) {
  %a = fsub double %x, %y
  ret double %a
}

; CHECK-LABEL: fmul64:
; CHECK: mul (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define double @fmul64(double %x, double %y) {
  %a = fmul double %x, %y
  ret double %a
}

; CHECK-LABEL: fdiv64:
; CHECK: div (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define double @fdiv64(double %x, double %y) {
  %a = fdiv double %x, %y
  ret double %a
}

; CHECK-LABEL: fabs64:
; CHECK: abs (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @fabs64(double %x) {
  %a = call double @llvm.fabs.f64(double %x)
  ret double %a
}

; CHECK-LABEL: fneg64:
; CHECK: neg (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @fneg64(double %x) {
  %a = fsub double -0., %x
  ret double %a
}

; CHECK-LABEL: copysign64:
; CHECK: copysign (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define double @copysign64(double %x, double %y) {
  %a = call double @llvm.copysign.f64(double %x, double %y)
  ret double %a
}

; CHECK-LABEL: sqrt64:
; CHECK: sqrt (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @sqrt64(double %x) {
  %a = call double @llvm.sqrt.f64(double %x)
  ret double %a
}

; CHECK-LABEL: ceil64:
; CHECK: ceil (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @ceil64(double %x) {
  %a = call double @llvm.ceil.f64(double %x)
  ret double %a
}

; CHECK-LABEL: floor64:
; CHECK: floor (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @floor64(double %x) {
  %a = call double @llvm.floor.f64(double %x)
  ret double %a
}

; CHECK-LABEL: trunc64:
; CHECK: trunc (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @trunc64(double %x) {
  %a = call double @llvm.trunc.f64(double %x)
  ret double %a
}

; CHECK-LABEL: nearest64:
; CHECK: nearest (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @nearest64(double %x) {
  %a = call double @llvm.nearbyint.f64(double %x)
  ret double %a
}

; CHECK-LABEL: nearest64_via_rint:
; CHECK: nearest (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @nearest64_via_rint(double %x) {
  %a = call double @llvm.rint.f64(double %x)
  ret double %a
}
