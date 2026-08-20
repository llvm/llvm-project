; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_60 | FileCheck %s
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_53 -mattr=+ptx65 | FileCheck %s --check-prefix=CHECK-F16
; RUN: llc < %s -mtriple=nvptx64 -mcpu=sm_80 -mattr=+ptx70 | FileCheck %s --check-prefix=CHECK-BF16

; fneg with bitcast (non-canonicalizing) user -> xor (expanded)
; CHECK-LABEL: test_fneg_bitcast(
define i32 @test_fneg_bitcast(float %a) {
; CHECK: xor.b32
  %neg = fneg float %a
  %b = bitcast float %neg to i32
  ret i32 %b
}

; fneg with canonicalizing user (fadd) -> folds into the sub
; CHECK-LABEL: test_fneg_canon(
define float @test_fneg_canon(float %a, float %b) {
; CHECK: sub.rn.f32
; CHECK-NOT: xor.b32
  %neg = fneg float %a
  %r = fadd float %neg, %b
  ret float %r
}

; fneg with canonicalized input (may still be a NaN) -> xor (expanded)
; CHECK-LABEL: test_fneg_canon_input(
define float @test_fneg_canon_input(float %a, float %b) {
; CHECK: xor.b32
; CHECK-NOT: neg.f32
  %add = fadd float %a, %b
  %neg = fneg float %add
  ret float %neg
}

; fneg f64 with canonicalizing user -> folds into the sub
; CHECK-LABEL: test_fneg_f64_canon(
define double @test_fneg_f64_canon(double %a, double %b) {
; CHECK: sub.rn.f64
; CHECK-NOT: xor.b64
  %neg = fneg double %a
  %r = fadd double %neg, %b
  ret double %r
}

; fneg f64 with bitcast (non-canonicalizing) user -> xor (expanded)
; CHECK-LABEL: test_fneg_f64_bitcast(
define i64 @test_fneg_f64_bitcast(double %a) {
; CHECK: xor.b64
  %neg = fneg double %a
  %b = bitcast double %neg to i64
  ret i64 %b
}

; fneg with mixed users (fadd + bitcast) -> xor (conservative)
; CHECK-LABEL: test_fneg_mixed(
define {i32, float} @test_fneg_mixed(float %a, float %b) {
; CHECK: xor.b32
; CHECK-NOT: neg.f32
  %neg = fneg float %a
  %cast = bitcast float %neg to i32
  %add = fadd float %neg, %b
  %r0 = insertvalue {i32, float} poison, i32 %cast, 0
  %r1 = insertvalue {i32, float} %r0, float %add, 1
  ret {i32, float} %r1
}

; fabs with bitcast (non-canonicalizing) user -> and (expanded)
; CHECK-LABEL: test_fabs_bitcast(
define i32 @test_fabs_bitcast(float %a) {
; CHECK: and.b32
; CHECK-NOT: abs.f32
  %abs = call float @llvm.fabs.f32(float %a)
  %b = bitcast float %abs to i32
  ret i32 %b
}

; fabs with canonicalizing user (fadd) -> native abs
; CHECK-LABEL: test_fabs_canon(
define float @test_fabs_canon(float %a, float %b) {
; CHECK: abs.f32
; CHECK-NOT: and.b32
  %abs = call float @llvm.fabs.f32(float %a)
  %r = fadd float %abs, %b
  ret float %r
}

; fabs with canonicalized input (may still be a NaN) -> and (expanded)
; CHECK-LABEL: test_fabs_canon_input(
define float @test_fabs_canon_input(float %a, float %b) {
; CHECK: and.b32
; CHECK-NOT: abs.f32
  %add = fadd float %a, %b
  %abs = call float @llvm.fabs.f32(float %add)
  ret float %abs
}

; fabs with setcc (canonicalizing) user -> native abs
; CHECK-LABEL: test_fabs_cmp(
define i1 @test_fabs_cmp(float %a) {
; CHECK: abs.f32
; CHECK-NOT: and.b32
  %abs = call float @llvm.fabs.f32(float %a)
  %r = fcmp olt float %abs, 1.0
  ret i1 %r
}

; fabs with mixed users (fadd + bitcast) -> and (conservative)
; CHECK-LABEL: test_fabs_mixed(
define {i32, float} @test_fabs_mixed(float %a, float %b) {
; CHECK: and.b32
; CHECK-NOT: abs.f32
  %abs = call float @llvm.fabs.f32(float %a)
  %cast = bitcast float %abs to i32
  %add = fadd float %abs, %b
  %r0 = insertvalue {i32, float} poison, i32 %cast, 0
  %r1 = insertvalue {i32, float} %r0, float %add, 1
  ret {i32, float} %r1
}

; fabs f64 with canonicalizing user -> native abs
; CHECK-LABEL: test_fabs_f64_canon(
define double @test_fabs_f64_canon(double %a, double %b) {
; CHECK: abs.f64
; CHECK-NOT: and.b64
  %abs = call double @llvm.fabs.f64(double %a)
  %r = fadd double %abs, %b
  ret double %r
}

; fabs f64 with non-canonicalizing user -> and (expanded)
; CHECK-LABEL: test_fabs_f64_bitcast(
define i64 @test_fabs_f64_bitcast(double %a) {
; CHECK: and.b64
; CHECK-NOT: abs.f64
  %abs = call double @llvm.fabs.f64(double %a)
  %b = bitcast double %abs to i64
  ret i64 %b
}

; f16 fabs with canonicalizing user -> native abs
; CHECK-F16-LABEL: test_fabs_f16_canon(
define half @test_fabs_f16_canon(half %a, half %b) {
; CHECK-F16: abs.f16
; CHECK-F16-NOT: and.b16
  %abs = call half @llvm.fabs.f16(half %a)
  %r = fadd half %abs, %b
  ret half %r
}

; f16 fabs with bitcast user -> expanded (and.b16, not native abs)
; CHECK-F16-LABEL: test_fabs_f16_bitcast(
define i16 @test_fabs_f16_bitcast(half %a) {
; CHECK-F16-NOT: abs.f16
; CHECK-F16: and.b16
  %abs = call half @llvm.fabs.f16(half %a)
  %b = bitcast half %abs to i16
  ret i16 %b
}

; fneg feeding fma -> native neg, not xor
; CHECK-LABEL: test_fneg_fma(
define float @test_fneg_fma(float %a, float %b, float %c) {
; CHECK: neg.f32
; CHECK: fma.rn.f32
  %neg = fneg float %a
  %r = call float @llvm.fma.f32(float %neg, float %b, float %c)
  ret float %r
}

; bf16 fabs with canonicalizing user -> native abs
; CHECK-BF16-LABEL: test_fabs_bf16_canon(
define bfloat @test_fabs_bf16_canon(bfloat %a, bfloat %b) {
; CHECK-BF16: abs.bf16
; CHECK-BF16-NOT: and.b16
  %abs = call bfloat @llvm.fabs.bf16(bfloat %a)
  %r = fadd bfloat %abs, %b
  ret bfloat %r
}

; bf16 fneg with canonicalized input -> xor (expanded)
; CHECK-BF16-LABEL: test_fneg_bf16_canon_input(
define bfloat @test_fneg_bf16_canon_input(bfloat %a, bfloat %b) {
; CHECK-BF16: xor.b16
; CHECK-BF16-NOT: neg.bf16
  %add = fadd bfloat %a, %b
  %neg = fneg bfloat %add
  ret bfloat %neg
}

; f16 fneg with canonicalized input -> xor (expanded)
; CHECK-F16-LABEL: test_fneg_f16_canon_input(
define half @test_fneg_f16_canon_input(half %a, half %b) {
; CHECK-F16: xor.b16
; CHECK-F16-NOT: neg.f16
  %add = fadd half %a, %b
  %neg = fneg half %add
  ret half %neg
}

; the packed types are handled the same way, elementwise
; CHECK-F16-LABEL: test_fabs_v2f16_canon(
define <2 x half> @test_fabs_v2f16_canon(<2 x half> %a, <2 x half> %b) {
; CHECK-F16: abs.f16x2
; CHECK-F16-NOT: and.b32
  %abs = call <2 x half> @llvm.fabs.v2f16(<2 x half> %a)
  %r = fadd <2 x half> %abs, %b
  ret <2 x half> %r
}

; CHECK-BF16-LABEL: test_fneg_v2bf16_canon_input(
define <2 x bfloat> @test_fneg_v2bf16_canon_input(<2 x bfloat> %a, <2 x bfloat> %b) {
; CHECK-BF16: xor.b32
; CHECK-BF16-NOT: neg.bf16x2
  %add = fadd <2 x bfloat> %a, %b
  %neg = fneg <2 x bfloat> %add
  ret <2 x bfloat> %neg
}

; fabs whose input is int-to-fp (never NaN) -> native abs
; CHECK-LABEL: test_fabs_sitofp_input(
define float @test_fabs_sitofp_input(i32 %i) {
; CHECK: abs.f32
; CHECK-NOT: and.b32
  %f = sitofp i32 %i to float
  %a = call float @llvm.fabs.f32(float %f)
  ret float %a
}

; fabs whose only user is a saturating fp-to-int (payload unobservable) -> native abs
; CHECK-LABEL: test_fabs_fptosi_sat(
define i32 @test_fabs_fptosi_sat(float %x) {
; CHECK: abs.f32
; CHECK-NOT: and.b32
  %a = call float @llvm.fabs.f32(float %x)
  %i = call i32 @llvm.fptosi.sat.i32.f32(float %a)
  ret i32 %i
}

; fabs whose only user is fp-to-uint (payload unobservable) -> native abs
; CHECK-LABEL: test_fabs_fptoui(
define i32 @test_fabs_fptoui(float %x) {
; CHECK: abs.f32
; CHECK-NOT: and.b32
  %a = call float @llvm.fabs.f32(float %x)
  %i = fptoui float %a to i32
  ret i32 %i
}
