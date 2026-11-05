; RUN: split-file %s %t
; RUN: llc -mtriple=nvptx64 -mcpu=sm_80 < %t/afn.ll | FileCheck %s
; RUN: not --crash llc -mtriple=nvptx64 -filetype=null %t/no-afn.ll 2>&1 | FileCheck %s --check-prefix=NO-AFN

;--- afn.ll
declare float @llvm.exp.f32(float)

define float @exp_afn(float %x) {
; CHECK-LABEL: exp_afn(
; CHECK:      mul.rn.f32 [[SCALED:%r[0-9]+]], {{%r[0-9]+}}, 0f3FB8AA3B;
; CHECK-NEXT: ex2.approx.f32 {{%r[0-9]+}}, [[SCALED]];
  %result = call afn float @llvm.exp.f32(float %x)
  ret float %result
}

define float @exp_afn_ftz(float %x) denormal_fpenv(preservesign) {
; CHECK-LABEL: exp_afn_ftz(
; CHECK:      mul.rn.ftz.f32 [[SCALED:%r[0-9]+]], {{%r[0-9]+}}, 0f3FB8AA3B;
; CHECK-NEXT: ex2.approx.ftz.f32 {{%r[0-9]+}}, [[SCALED]];
  %result = call afn float @llvm.exp.f32(float %x)
  ret float %result
}

;--- no-afn.ll
declare float @llvm.exp.f32(float)

; NO-AFN: error: no libcall available for fexp
define float @exp_no_afn(float %x) {
  %result = call float @llvm.exp.f32(float %x)
  ret float %result
}
