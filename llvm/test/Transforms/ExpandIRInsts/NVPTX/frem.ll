; REQUIRES: nvptx-registered-target
; RUN: opt -mtriple=nvptx64 -passes="require<libcall-lowering-info>,expand-ir-insts<O0>" %s -S | FileCheck %s --implicit-check-not=" = frem "
; RUN: opt -mtriple=nvptx64 -passes="require<libcall-lowering-info>,expand-ir-insts<O2>" %s -S | FileCheck %s --implicit-check-not=" = frem "

; Half and bfloat remainders use float arithmetic, processing 11 and 8 bits
; per iteration respectively. The result keeps the numerator's sign.
define half @frem_f16(half %x, half %y) {
; CHECK-LABEL: define half @frem_f16(
; CHECK: fpext half {{%.*}} to float
; CHECK: call { float, i32 } @llvm.frexp.f32.i32
; CHECK: call float @llvm.ldexp.f32.i32(float {{%.*}}, i32 11)
; CHECK: call half @llvm.copysign.f16(half 0.000000e+00, half %x)
; CHECK: call float @llvm.fma.f32
; CHECK: fptrunc float {{%.*}} to half
; CHECK: call half @llvm.copysign.f16(half {{%.*}}, half %x)
  %r = frem half %x, %y
  ret half %r
}

define bfloat @frem_bf16(bfloat %x, bfloat %y) {
; CHECK-LABEL: define bfloat @frem_bf16(
; CHECK: fpext bfloat {{%.*}} to float
; CHECK: call { float, i32 } @llvm.frexp.f32.i32
; CHECK: call float @llvm.ldexp.f32.i32(float {{%.*}}, i32 8)
; CHECK: call bfloat @llvm.copysign.bf16(bfloat 0.000000e+00, bfloat %x)
; CHECK: call float @llvm.fma.f32
; CHECK: fptrunc float {{%.*}} to bfloat
; CHECK: call bfloat @llvm.copysign.bf16(bfloat {{%.*}}, bfloat %x)
  %r = frem bfloat %x, %y
  ret bfloat %r
}

; Preserve NaNs, reject zero denominators and infinite numerators, and copy
; the numerator's sign onto zero as well as nonzero remainders.
define float @frem_f32(float %x, float %y) {
; CHECK-LABEL: define float @frem_f32(
; CHECK: [[BADY:%.*]] = fcmp ueq float %y, 0.000000e+00
; CHECK-NEXT: [[NANRET:%.*]] = select i1 [[BADY]], float +qnan, float {{%.*}}
; CHECK: [[ABSX:%.*]] = call float @llvm.fabs.f32(float %x)
; CHECK-NEXT: [[FINITEX:%.*]] = fcmp ult float [[ABSX]], +inf
; CHECK-NEXT: %r = select i1 [[FINITEX]], float [[NANRET]], float +qnan
; CHECK: call float @llvm.ldexp.f32.i32(float {{%.*}}, i32 12)
; CHECK: call float @llvm.copysign.f32(float 0.000000e+00, float %x)
; CHECK: call float @llvm.fma.f32
; CHECK: call float @llvm.copysign.f32(float {{%.*}}, float %x)
  %r = frem float %x, %y
  ret float %r
}

define double @frem_f64(double %x, double %y) {
; CHECK-LABEL: define double @frem_f64(
; CHECK: call { double, i32 } @llvm.frexp.f64.i32
; CHECK: call double @llvm.ldexp.f64.i32(double {{%.*}}, i32 26)
; CHECK: call double @llvm.copysign.f64(double 0.000000e+00, double %x)
; CHECK: call double @llvm.fma.f64
; CHECK: call double @llvm.copysign.f64(double {{%.*}}, double %x)
  %r = frem double %x, %y
  ret double %r
}

; Other fast-math flags do not select the approximation without afn.
define float @frem_other_flags(float %x, float %y) {
; CHECK-LABEL: define float @frem_other_flags(
; CHECK: call {{.*}}{ float, i32 } @llvm.frexp.f32.i32
; CHECK: call {{.*}}float @llvm.ldexp.f32.i32(float {{%.*}}, i32 12)
  %r = frem reassoc nnan ninf nsz arcp contract float %x, %y
  ret float %r
}

; afn selects the straight-line approximation. Its division does not inherit
; afn, and the aggregate fast flag selects the same expansion.
define float @frem_afn(float %x, float %y) {
; CHECK-LABEL: define float @frem_afn(
; CHECK-NEXT: [[QUOT:%.*]] = fdiv float %x, %y
; CHECK-NEXT: [[TRUNC:%.*]] = call float @llvm.trunc.f32(float [[QUOT]])
; CHECK-NEXT: [[NEG:%.*]] = fneg float [[TRUNC]]
; CHECK-NEXT: %r = call float @llvm.fma.f32(float [[NEG]], float %y, float %x)
; CHECK-NEXT: ret float %r
  %r = frem afn float %x, %y
  ret float %r
}

define float @frem_fast(float %x, float %y) {
; CHECK-LABEL: define float @frem_fast(
; CHECK-NEXT: [[QUOT:%.*]] = fdiv float %x, %y
; CHECK-NEXT: [[TRUNC:%.*]] = call float @llvm.trunc.f32(float [[QUOT]])
; CHECK-NEXT: [[NEG:%.*]] = fneg float [[TRUNC]]
; CHECK-NEXT: %r = call float @llvm.fma.f32(float [[NEG]], float %y, float %x)
; CHECK-NEXT: ret float %r
  %r = frem fast float %x, %y
  ret float %r
}
