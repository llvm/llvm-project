; RUN: llc < %s -asm-verbose=false -disable-wasm-fallthrough-return-opt -wasm-keep-registers | FileCheck %s

; Test that basic 32-bit floating-point operations assemble as expected.

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
; CHECK-NEXT: .functype fadd32 (f32, f32) -> (f32){{$}}
; CHECK-NEXT: local.get $push[[L0:[0-9]+]]=, 0{{$}}
; CHECK-NEXT: local.get $push[[L1:[0-9]+]]=, 1{{$}}
; CHECK-NEXT: f32.add $push[[LR:[0-9]+]]=, $pop[[L0]], $pop[[L1]]{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @fadd32(float %x, float %y) {
  %a = fadd float %x, %y
  ret float %a
}

; CHECK-LABEL: fsub32:
; CHECK: f32.sub $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @fsub32(float %x, float %y) {
  %a = fsub float %x, %y
  ret float %a
}

; CHECK-LABEL: fmul32:
; CHECK: f32.mul $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @fmul32(float %x, float %y) {
  %a = fmul float %x, %y
  ret float %a
}

; CHECK-LABEL: fdiv32:
; CHECK: f32.div $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @fdiv32(float %x, float %y) {
  %a = fdiv float %x, %y
  ret float %a
}

; CHECK-LABEL: fabs32:
; CHECK: f32.abs $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @fabs32(float %x) {
  %a = call float @llvm.fabs.f32(float %x)
  ret float %a
}

; CHECK-LABEL: fneg32:
; CHECK: f32.neg $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @fneg32(float %x) {
  %a = fsub float -0., %x
  ret float %a
}

; CHECK-LABEL: copysign32:
; CHECK: f32.copysign $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @copysign32(float %x, float %y) {
  %a = call float @llvm.copysign.f32(float %x, float %y)
  ret float %a
}

; CHECK-LABEL: sqrt32:
; CHECK: f32.sqrt $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @sqrt32(float %x) {
  %a = call float @llvm.sqrt.f32(float %x)
  ret float %a
}

; CHECK-LABEL: ceil32:
; CHECK: f32.ceil $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @ceil32(float %x) {
  %a = call float @llvm.ceil.f32(float %x)
  ret float %a
}

; CHECK-LABEL: floor32:
; CHECK: f32.floor $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @floor32(float %x) {
  %a = call float @llvm.floor.f32(float %x)
  ret float %a
}

; CHECK-LABEL: trunc32:
; CHECK: f32.trunc $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @trunc32(float %x) {
  %a = call float @llvm.trunc.f32(float %x)
  ret float %a
}

; CHECK-LABEL: nearest32:
; CHECK: f32.nearest $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @nearest32(float %x) {
  %a = call float @llvm.nearbyint.f32(float %x)
  ret float %a
}

; CHECK-LABEL: nearest32_via_rint:
; CHECK: f32.nearest $push[[LR:[0-9]+]]=, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @nearest32_via_rint(float %x) {
  %a = call float @llvm.rint.f32(float %x)
  ret float %a
}

; CHECK-LABEL: fmin32:
; CHECK: f32.min $push1=, $pop{{[0-9]+}}, $pop[[LR]]{{$}}
; CHECK-NEXT: return $pop1{{$}}
define float @fmin32(float %x) {
  %a = fcmp ult float %x, 0.0
  %b = select i1 %a, float %x, float 0.0
  ret float %b
}

; CHECK-LABEL: fmax32:
; CHECK: f32.max $push1=, $pop{{[0-9]+}}, $pop[[LR]]{{$}}
; CHECK-NEXT: return $pop1{{$}}
define float @fmax32(float %x) {
  %a = fcmp ugt float %x, 0.0
  %b = select i1 %a, float %x, float 0.0
  ret float %b
}

; CHECK-LABEL: fmin32_intrinsic:
; CHECK: f32.min $push0=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop0{{$}}
declare float @llvm.minimum.f32(float, float)
define float @fmin32_intrinsic(float %x, float %y) {
  %a = call float @llvm.minimum.f32(float %x, float %y)
  ret float %a
}

; CHECK-LABEL: fminnum32_intrinsic:
; CHECK: f32.min $push0=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop0{{$}}
declare float @llvm.minnum.f32(float, float)
define float @fminnum32_intrinsic(float %x, float %y) {
  %a = call nnan float @llvm.minnum.f32(float %x, float %y)
  ret float %a
}

; CHECK-LABEL: fmax32_intrinsic:
; CHECK: f32.max $push0=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop0{{$}}
declare float @llvm.maximum.f32(float, float)
define float @fmax32_intrinsic(float %x, float %y) {
  %a = call float @llvm.maximum.f32(float %x, float %y)
  ret float %a
}

; CHECK-LABEL: fmaxnum32_intrinsic:
; CHECK: f32.max $push0=, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop0{{$}}
declare float @llvm.maxnum.f32(float, float)
define float @fmaxnum32_intrinsic(float %x, float %y) {
  %a = call nnan float @llvm.maxnum.f32(float %x, float %y)
  ret float %a
}

; CHECK-LABEL: fma32:
; CHECK: {{^}} call $push[[LR:[0-9]+]]=, fmaf, $pop{{[0-9]+}}, $pop{{[0-9]+}}, $pop{{[0-9]+}}{{$}}
; CHECK-NEXT: return $pop[[LR]]{{$}}
define float @fma32(float %a, float %b, float %c) {
  %d = call float @llvm.fma.f32(float %a, float %b, float %c)
  ret float %d
}
