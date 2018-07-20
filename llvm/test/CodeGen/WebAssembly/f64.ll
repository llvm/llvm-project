; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt | FileCheck %s

; Test that basic 64-bit floating-point operations assemble as expected.

target datalayout = "e-m:e-p:32:32-i64:64-n32:64-S128"
target triple = "wasm32-unknown-unknown"

declare double @llvm.fabs.f64(double)
declare double @llvm.copysign.f64(double, double)
declare double @llvm.sqrt.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.trunc.f64(double)
declare double @llvm.nearbyint.f64(double)
declare double @llvm.rint.f64(double)
declare double @llvm.fma.f64(double, double, double)

; CHECK-LABEL: fadd64:
; CHECK-NEXT: .param f64, f64{{$}}
; CHECK-NEXT: .result f64{{$}}
; CHECK-NEXT: get_local $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: get_local $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f64.add $push[[LR:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @fadd64(double %x, double %y) {
  %a = fadd double %x, %y
  ret double %a
}

; CHECK-LABEL: fsub64:
; CHECK: f64.sub $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @fsub64(double %x, double %y) {
  %a = fsub double %x, %y
  ret double %a
}

; CHECK-LABEL: fmul64:
; CHECK: f64.mul $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @fmul64(double %x, double %y) {
  %a = fmul double %x, %y
  ret double %a
}

; CHECK-LABEL: fdiv64:
; CHECK: f64.div $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @fdiv64(double %x, double %y) {
  %a = fdiv double %x, %y
  ret double %a
}

; CHECK-LABEL: fabs64:
; CHECK: f64.abs $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @fabs64(double %x) {
  %a = call double @llvm.fabs.f64(double %x)
  ret double %a
}

; CHECK-LABEL: fneg64:
; CHECK: f64.neg $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @fneg64(double %x) {
  %a = fsub double -0., %x
  ret double %a
}

; CHECK-LABEL: copysign64:
; CHECK: f64.copysign $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @copysign64(double %x, double %y) {
  %a = call double @llvm.copysign.f64(double %x, double %y)
  ret double %a
}

; CHECK-LABEL: sqrt64:
; CHECK: f64.sqrt $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @sqrt64(double %x) {
  %a = call double @llvm.sqrt.f64(double %x)
  ret double %a
}

; CHECK-LABEL: ceil64:
; CHECK: f64.ceil $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @ceil64(double %x) {
  %a = call double @llvm.ceil.f64(double %x)
  ret double %a
}

; CHECK-LABEL: floor64:
; CHECK: f64.floor $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @floor64(double %x) {
  %a = call double @llvm.floor.f64(double %x)
  ret double %a
}

; CHECK-LABEL: trunc64:
; CHECK: f64.trunc $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @trunc64(double %x) {
  %a = call double @llvm.trunc.f64(double %x)
  ret double %a
}

; CHECK-LABEL: nearest64:
; CHECK: f64.nearest $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @nearest64(double %x) {
  %a = call double @llvm.nearbyint.f64(double %x)
  ret double %a
}

; CHECK-LABEL: nearest64_via_rint:
; CHECK: f64.nearest $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
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
; CHECK: f64.min $push1=, $pop{{[0-9]+}}, $pop[[LR]]{{$}}
; CHECK-NEXT: return $pop1{{$}}
define double @fmin64(double %x) {
  %a = fcmp ult double %x, 0.0
  %b = select i1 %a, double %x, double 0.0
  ret double %b
}

; CHECK-LABEL: fmax64:
; CHECK: f64.max $push1=, $pop{{[0-9]+}}, $pop[[LR]]{{$}}
; CHECK-NEXT: return $pop1{{$}}
define double @fmax64(double %x) {
  %a = fcmp ugt double %x, 0.0
  %b = select i1 %a, double %x, double 0.0
  ret double %b
}

; CHECK-LABEL: fma64:
; CHECK: {{^}} f64.call $push[[LR:[0-9]+]]=, fma@FUNCTION, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define double @fma64(double %a, double %b, double %c) {
  %d = call double @llvm.fma.f64(double %a, double %b, double %c)
  ret double %d
}
