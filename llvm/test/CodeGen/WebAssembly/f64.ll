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
; CHECK-NEXT: get_local push, 1{{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
; CHECK-NEXT: get_local push, 0{{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
; CHECK-NEXT: f64.add push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
; CHECK-NEXT: return (get_local 4){{$}}
define double @fadd64(double %x, double %y) {
  %a = fadd double %x, %y
  ret double %a
}

; CHECK-LABEL: fsub64:
; CHECK: f64.sub push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define double @fsub64(double %x, double %y) {
  %a = fsub double %x, %y
  ret double %a
}

; CHECK-LABEL: fmul64:
; CHECK: f64.mul push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define double @fmul64(double %x, double %y) {
  %a = fmul double %x, %y
  ret double %a
}

; CHECK-LABEL: fdiv64:
; CHECK: f64.div push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define double @fdiv64(double %x, double %y) {
  %a = fdiv double %x, %y
  ret double %a
}

; CHECK-LABEL: fabs64:
; CHECK: f64.abs push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @fabs64(double %x) {
  %a = call double @llvm.fabs.f64(double %x)
  ret double %a
}

; CHECK-LABEL: fneg64:
; CHECK: f64.neg push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @fneg64(double %x) {
  %a = fsub double -0., %x
  ret double %a
}

; CHECK-LABEL: copysign64:
; CHECK: f64.copysign push, (get_local 3), (get_local 2){{$}}
; CHECK-NEXT: set_local 4, pop{{$}}
define double @copysign64(double %x, double %y) {
  %a = call double @llvm.copysign.f64(double %x, double %y)
  ret double %a
}

; CHECK-LABEL: sqrt64:
; CHECK: f64.sqrt push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @sqrt64(double %x) {
  %a = call double @llvm.sqrt.f64(double %x)
  ret double %a
}

; CHECK-LABEL: ceil64:
; CHECK: f64.ceil push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @ceil64(double %x) {
  %a = call double @llvm.ceil.f64(double %x)
  ret double %a
}

; CHECK-LABEL: floor64:
; CHECK: f64.floor push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @floor64(double %x) {
  %a = call double @llvm.floor.f64(double %x)
  ret double %a
}

; CHECK-LABEL: trunc64:
; CHECK: f64.trunc push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @trunc64(double %x) {
  %a = call double @llvm.trunc.f64(double %x)
  ret double %a
}

; CHECK-LABEL: nearest64:
; CHECK: f64.nearest push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @nearest64(double %x) {
  %a = call double @llvm.nearbyint.f64(double %x)
  ret double %a
}

; CHECK-LABEL: nearest64_via_rint:
; CHECK: f64.nearest push, (get_local 1){{$}}
; CHECK-NEXT: set_local 2, pop{{$}}
define double @nearest64_via_rint(double %x) {
  %a = call double @llvm.rint.f64(double %x)
  ret double %a
}

; Min and max tests. LLVM currently only forms fminnan and fmaxnan nodes in
; cases where there's a single fcmp with a select and it can prove that one
; of the arms is never NaN, so we only test that case. In the future if LLVM
; learns to form fminnan/fmaxnan in more cases, we can write more general
; tests.

; CHECK-LABEL: fmin64:
; CHECK: f64.min push, (get_local 1), (get_local 2){{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
define double @fmin64(double %x) {
  %a = fcmp ult double %x, 0.0
  %b = select i1 %a, double %x, double 0.0
  ret double %b
}

; CHECK-LABEL: fmax64:
; CHECK: f64.max push, (get_local 1), (get_local 2){{$}}
; CHECK-NEXT: set_local 3, pop{{$}}
define double @fmax64(double %x) {
  %a = fcmp ugt double %x, 0.0
  %b = select i1 %a, double %x, double 0.0
  ret double %b
}
