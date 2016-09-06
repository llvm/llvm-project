; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon -aarch64-enable-simd-scalar| FileCheck %s

define <8 x i8> @add8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: add {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = add <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @add16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: add {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = add <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <4 x i16> @add4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: add {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = add <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @add8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: add {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = add <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <2 x i32> @add2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: add {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = add <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @add4x32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: add {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = add <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <2 x i64> @add2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: add {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = add <2 x i64> %A, %B;
	ret <2 x i64> %tmp3
}

define <2 x float> @add2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fadd {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = fadd <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @add4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fadd {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = fadd <4 x float> %A, %B;
	ret <4 x float> %tmp3
}
define <2 x double> @add2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: add {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = fadd <2 x double> %A, %B;
	ret <2 x double> %tmp3
}

define <8 x i8> @sub8xi8(<8 x i8> %A, <8 x i8> %B) {
;CHECK: sub {{v[0-9]+}}.8b, {{v[0-9]+}}.8b, {{v[0-9]+}}.8b
	%tmp3 = sub <8 x i8> %A, %B;
	ret <8 x i8> %tmp3
}

define <16 x i8> @sub16xi8(<16 x i8> %A, <16 x i8> %B) {
;CHECK: sub {{v[0-9]+}}.16b, {{v[0-9]+}}.16b, {{v[0-9]+}}.16b
	%tmp3 = sub <16 x i8> %A, %B;
	ret <16 x i8> %tmp3
}

define <4 x i16> @sub4xi16(<4 x i16> %A, <4 x i16> %B) {
;CHECK: sub {{v[0-9]+}}.4h, {{v[0-9]+}}.4h, {{v[0-9]+}}.4h
	%tmp3 = sub <4 x i16> %A, %B;
	ret <4 x i16> %tmp3
}

define <8 x i16> @sub8xi16(<8 x i16> %A, <8 x i16> %B) {
;CHECK: sub {{v[0-9]+}}.8h, {{v[0-9]+}}.8h, {{v[0-9]+}}.8h
	%tmp3 = sub <8 x i16> %A, %B;
	ret <8 x i16> %tmp3
}

define <2 x i32> @sub2xi32(<2 x i32> %A, <2 x i32> %B) {
;CHECK: sub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = sub <2 x i32> %A, %B;
	ret <2 x i32> %tmp3
}

define <4 x i32> @sub4x32(<4 x i32> %A, <4 x i32> %B) {
;CHECK: sub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = sub <4 x i32> %A, %B;
	ret <4 x i32> %tmp3
}

define <2 x i64> @sub2xi64(<2 x i64> %A, <2 x i64> %B) {
;CHECK: sub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = sub <2 x i64> %A, %B;
	ret <2 x i64> %tmp3
}

define <2 x float> @sub2xfloat(<2 x float> %A, <2 x float> %B) {
;CHECK: fsub {{v[0-9]+}}.2s, {{v[0-9]+}}.2s, {{v[0-9]+}}.2s
	%tmp3 = fsub <2 x float> %A, %B;
	ret <2 x float> %tmp3
}

define <4 x float> @sub4xfloat(<4 x float> %A, <4 x float> %B) {
;CHECK: fsub {{v[0-9]+}}.4s, {{v[0-9]+}}.4s, {{v[0-9]+}}.4s
	%tmp3 = fsub <4 x float> %A, %B;
	ret <4 x float> %tmp3
}
define <2 x double> @sub2xdouble(<2 x double> %A, <2 x double> %B) {
;CHECK: sub {{v[0-9]+}}.2d, {{v[0-9]+}}.2d, {{v[0-9]+}}.2d
	%tmp3 = fsub <2 x double> %A, %B;
	ret <2 x double> %tmp3
}

define <1 x double> @test_vadd_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vadd_f64
; CHECK: fadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fadd <1 x double> %a, %b
  ret <1 x double> %1
}

define <1 x double> @test_vmul_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vmul_f64
; CHECK: fmul d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fmul <1 x double> %a, %b
  ret <1 x double> %1
}

define <1 x double> @test_vdiv_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vdiv_f64
; CHECK: fdiv d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fdiv <1 x double> %a, %b
  ret <1 x double> %1
}

define <1 x double> @test_vmla_f64(<1 x double> %a, <1 x double> %b, <1 x double> %c) {
; CHECK-LABEL: test_vmla_f64
; CHECK: fmul d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: fadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fmul <1 x double> %b, %c
  %2 = fadd <1 x double> %1, %a
  ret <1 x double> %2
}

define <1 x double> @test_vmls_f64(<1 x double> %a, <1 x double> %b, <1 x double> %c) {
; CHECK-LABEL: test_vmls_f64
; CHECK: fmul d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: fsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fmul <1 x double> %b, %c
  %2 = fsub <1 x double> %a, %1
  ret <1 x double> %2
}

define <1 x double> @test_vfms_f64(<1 x double> %a, <1 x double> %b, <1 x double> %c) {
; CHECK-LABEL: test_vfms_f64
; CHECK: fmsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fsub <1 x double> <double -0.000000e+00>, %b
  %2 = tail call <1 x double> @llvm.fma.v1f64(<1 x double> %1, <1 x double> %c, <1 x double> %a)
  ret <1 x double> %2
}

define <1 x double> @test_vfma_f64(<1 x double> %a, <1 x double> %b, <1 x double> %c) {
; CHECK-LABEL: test_vfma_f64
; CHECK: fmadd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.fma.v1f64(<1 x double> %b, <1 x double> %c, <1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vsub_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vsub_f64
; CHECK: fsub d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fsub <1 x double> %a, %b
  ret <1 x double> %1
}

define <1 x double> @test_vabd_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vabd_f64
; CHECK: fabd d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.aarch64.neon.fabd.v1f64(<1 x double> %a, <1 x double> %b)
  ret <1 x double> %1
}

define <1 x double> @test_vmax_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vmax_f64
; CHECK: fmax d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.aarch64.neon.fmax.v1f64(<1 x double> %a, <1 x double> %b)
  ret <1 x double> %1
}

define <1 x double> @test_vmin_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vmin_f64
; CHECK: fmin d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.aarch64.neon.fmin.v1f64(<1 x double> %a, <1 x double> %b)
  ret <1 x double> %1
}

define <1 x double> @test_vmaxnm_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vmaxnm_f64
; CHECK: fmaxnm d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.aarch64.neon.fmaxnm.v1f64(<1 x double> %a, <1 x double> %b)
  ret <1 x double> %1
}

define <1 x double> @test_vminnm_f64(<1 x double> %a, <1 x double> %b) {
; CHECK-LABEL: test_vminnm_f64
; CHECK: fminnm d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.aarch64.neon.fminnm.v1f64(<1 x double> %a, <1 x double> %b)
  ret <1 x double> %1
}

define <1 x double> @test_vabs_f64(<1 x double> %a) {
; CHECK-LABEL: test_vabs_f64
; CHECK: fabs d{{[0-9]+}}, d{{[0-9]+}}
  %1 = tail call <1 x double> @llvm.fabs.v1f64(<1 x double> %a)
  ret <1 x double> %1
}

define <1 x double> @test_vneg_f64(<1 x double> %a) {
; CHECK-LABEL: test_vneg_f64
; CHECK: fneg d{{[0-9]+}}, d{{[0-9]+}}
  %1 = fsub <1 x double> <double -0.000000e+00>, %a
  ret <1 x double> %1
}

declare <1 x double> @llvm.fabs.v1f64(<1 x double>)
declare <1 x double> @llvm.aarch64.neon.fminnm.v1f64(<1 x double>, <1 x double>)
declare <1 x double> @llvm.aarch64.neon.fmaxnm.v1f64(<1 x double>, <1 x double>)
declare <1 x double> @llvm.aarch64.neon.fmin.v1f64(<1 x double>, <1 x double>)
declare <1 x double> @llvm.aarch64.neon.fmax.v1f64(<1 x double>, <1 x double>)
declare <1 x double> @llvm.aarch64.neon.fabd.v1f64(<1 x double>, <1 x double>)
declare <1 x double> @llvm.fma.v1f64(<1 x double>, <1 x double>, <1 x double>)
