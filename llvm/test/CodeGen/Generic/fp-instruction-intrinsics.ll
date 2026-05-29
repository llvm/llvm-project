; REQUIRES: x86-registered-target
; RUN: llc -O0 -mtriple=x86_64-pc-linux < %s | FileCheck %s

; Verify that each new FP instruction intrinsic lowers to the expected
; machine instruction on x86_64.

; llvm.fadd lowers to addsd (double SSE add)
define double @test_fadd(double %a, double %b) {
; CHECK-LABEL: test_fadd:
; CHECK: addsd
  %r = call double @llvm.fadd.f64(double %a, double %b)
  ret double %r
}

; llvm.fsub lowers to subsd
define double @test_fsub(double %a, double %b) {
; CHECK-LABEL: test_fsub:
; CHECK: subsd
  %r = call double @llvm.fsub.f64(double %a, double %b)
  ret double %r
}

; llvm.fmul lowers to mulsd
define double @test_fmul(double %a, double %b) {
; CHECK-LABEL: test_fmul:
; CHECK: mulsd
  %r = call double @llvm.fmul.f64(double %a, double %b)
  ret double %r
}

; llvm.fdiv lowers to divsd
define double @test_fdiv(double %a, double %b) {
; CHECK-LABEL: test_fdiv:
; CHECK: divsd
  %r = call double @llvm.fdiv.f64(double %a, double %b)
  ret double %r
}

; llvm.fptrunc lowers to cvtsd2ss
define float @test_fptrunc(double %a) {
; CHECK-LABEL: test_fptrunc:
; CHECK: cvtsd2ss
  %r = call float @llvm.fptrunc.f32.f64(double %a)
  ret float %r
}

; llvm.fpext lowers to cvtss2sd
define double @test_fpext(float %a) {
; CHECK-LABEL: test_fpext:
; CHECK: cvtss2sd
  %r = call double @llvm.fpext.f64.f32(float %a)
  ret double %r
}

; llvm.sitofp lowers to cvtsi2ss
define float @test_sitofp(i32 %a) {
; CHECK-LABEL: test_sitofp:
; CHECK: cvtsi2ss
  %r = call float @llvm.sitofp.f32.i32(i32 %a)
  ret float %r
}

; llvm.fptosi lowers to cvttss2si
define i32 @test_fptosi(float %a) {
; CHECK-LABEL: test_fptosi:
; CHECK: cvttss2si
  %r = call i32 @llvm.fptosi.i32.f32(float %a)
  ret i32 %r
}

; llvm.fcmp (quiet compare) lowers to cmpeqss (SSE mask form for i1 result)
define i1 @test_fcmp_oeq(float %a, float %b) {
; CHECK-LABEL: test_fcmp_oeq:
; CHECK: cmpeqss
  %r = call i1 @llvm.fcmp.f32(float %a, float %b, metadata !"oeq")
  ret i1 %r
}

declare double @llvm.fadd.f64(double, double)
declare double @llvm.fsub.f64(double, double)
declare double @llvm.fmul.f64(double, double)
declare double @llvm.fdiv.f64(double, double)
declare float @llvm.fptrunc.f32.f64(double)
declare double @llvm.fpext.f64.f32(float)
declare float @llvm.sitofp.f32.i32(i32)
declare i32 @llvm.fptosi.i32.f32(float)
declare i1 @llvm.fcmp.f32(float, float, metadata)
