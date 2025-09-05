; RUN: llc -mtriple=xcore-unknown-unknown < %s | FileCheck %s

define { half, i32 } @test_frexp_f16_i32(half %a) nounwind {
; CHECK-LABEL: test_frexp_f16_i32:
; CHECK: bl __extendhfsf2
; CHECK: bl frexpf
; CHECK: ldw r{{[0-9]+}}, sp[1]
; CHECK: bl __truncsfhf2
%result = call { half, i32 } @llvm.frexp.f16.i32(half %a)
  ret { half, i32 } %result
}

define { <2 x half>, <2 x i32> } @test_frexp_v2f16_v2i32(<2 x half> %a) nounwind {
; CHECK-LABEL: test_frexp_v2f16_v2i32:
; CHECK: bl frexpf
; CHECK: bl frexpf
  %result = call { <2 x half>, <2 x i32> } @llvm.frexp.v2f16.v2i32(<2 x half> %a)
  ret { <2 x half>, <2 x i32> } %result
}

define { float, i32 } @test_frexp_f32_i32(float %a) nounwind {
; CHECK-LABEL: test_frexp_f32_i32:
; CHECK: bl frexpf
  %result = call { float, i32 } @llvm.frexp.f32.i32(float %a)
  ret { float, i32 } %result
}

define { float, i32 } @test_frexp_f32_i32_tailcall(float %a) nounwind {
; CHECK-LABEL: test_frexp_f32_i32_tailcall:
; CHECK: bl frexpf
  %result = tail call { float, i32 } @llvm.frexp.f32.i32(float %a)
  ret { float, i32 } %result
}

define { <2 x float>, <2 x i32> } @test_frexp_v2f32_v2i32(<2 x float> %a) nounwind {
; CHECK-LABEL: test_frexp_v2f32_v2i32:
; CHECK: bl frexpf
; CHECK: bl frexpf
  %result = call { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float> %a)
  ret { <2 x float>, <2 x i32> } %result
}

define { double, i32 } @test_frexp_f64_i32(double %a) nounwind {
; CHECK-LABEL: test_frexp_f64_i32:
; CHECK: bl frexp
  %result = call { double, i32 } @llvm.frexp.f64.i32(double %a)
  ret { double, i32 } %result
}

define { <2 x double>, <2 x i32> } @test_frexp_v2f64_v2i32(<2 x double> %a) nounwind {
; CHECK-LABEL: test_frexp_v2f64_v2i32:
; CHECK: bl frexp
; CHECK: bl frexp
  %result = call { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double> %a)
  ret { <2 x double>, <2 x i32> } %result
}

define { fp128, i32 } @test_frexp_fp128_i32(fp128 %a) nounwind {
; CHECK-LABEL: test_frexp_fp128_i32:
; CHECK: bl frexpl
  %result = call { fp128, i32 } @llvm.frexp.fp128.i32(fp128 %a)
  ret { fp128, i32 } %result
}

define { <2 x fp128>, <2 x i32> } @test_frexp_v2fp128_v2i32(<2 x fp128> %a) nounwind {
; CHECK-LABEL: test_frexp_v2fp128_v2i32:
; CHECK: bl frexpl
; CHECK: bl frexpl
  %result = call { <2 x fp128>, <2 x i32> } @llvm.frexp.v2fp128.v2i32(<2 x fp128> %a)
  ret { <2 x fp128>, <2 x i32> } %result
}

declare { half, i32 } @llvm.frexp.f16.i32(half) #0
declare { <2 x half>, <2 x i32> } @llvm.frexp.v2f16.v2i32(<2 x half>) #0

declare { float, i32 } @llvm.frexp.f32.i32(float) #0
declare { <2 x float>, <2 x i32> } @llvm.frexp.v2f32.v2i32(<2 x float>) #0

declare { double, i32 } @llvm.frexp.f64.i32(double) #0
declare { <2 x double>, <2 x i32> } @llvm.frexp.v2f64.v2i32(<2 x double>) #0

declare { fp128, i32 } @llvm.frexp.fp128.i32(fp128) #0
declare { <2 x fp128>, <2 x i32> } @llvm.frexp.v2fp128.v2i32(<2 x fp128>) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
