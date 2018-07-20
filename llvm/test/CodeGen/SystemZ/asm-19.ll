; Test the vector constraint "v" and explicit vector register names.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 -no-integrated-as | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 -no-integrated-as | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-Z14

define float @f1() {
; CHECK-LABEL: f1:
; CHECK: lzer %f1
; CHECK: blah %f0 %f1
; CHECK: br %r14
  %val = call float asm "blah $0 $1", "=&v,v" (float 0.0)
  ret float %val
}

define double @f2() {
; CHECK-LABEL: f2:
; CHECK: lzdr %f1
; CHECK: blah %f0 %f1
; CHECK: br %r14
  %val = call double asm "blah $0 $1", "=&v,v" (double 0.0)
  ret double %val
}

define fp128 @f3() {
; CHECK-LABEL: f3:
; CHECK-Z14: vzero %v0
; CHECK: blah %v1 %v0
; CHECK: vst %v1, 0(%r2)
; CHECK: br %r14
  %val = call fp128 asm "blah $0 $1", "=&v,v" (fp128 0xL00000000000000000000000000000000)
  ret fp128 %val
}

define <2 x i64> @f4() {
; CHECK-LABEL: f4:
; CHECK: vrepig  %v0, 1
; CHECK: blah %v24 %v0
; CHECK: br %r14
  %val = call <2 x i64> asm "blah $0 $1", "=&v,v" (<2 x i64> <i64 1, i64 1>)
  ret <2 x i64> %val
}

define <4 x i32> @f5() {
; CHECK-LABEL: f5:
; CHECK: vrepif  %v0, 1
; CHECK: blah %v24 %v0
; CHECK: br %r14
  %val = call <4 x i32> asm "blah $0 $1", "=&v,v" (<4 x i32> <i32 1, i32 1, i32 1, i32 1>)
  ret <4 x i32> %val
}

define <8 x i16> @f6() {
; CHECK-LABEL: f6:
; CHECK: vrepih  %v0, 1
; CHECK: blah %v24 %v0
; CHECK: br %r14
  %val = call <8 x i16> asm "blah $0 $1", "=&v,v" (<8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
  ret <8 x i16> %val
}

define <16 x i8> @f7() {
; CHECK-LABEL: f7:
; CHECK: vrepib  %v0, 1
; CHECK: blah %v24 %v0
; CHECK: br %r14
  %val = call <16 x i8> asm "blah $0 $1", "=&v,v" (<16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1,
                                                              i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>)
  ret <16 x i8> %val
}

define <2 x double> @f8() {
; CHECK-LABEL: f8:
; CHECK: vgbm  %v0, 0
; CHECK: blah %v24 %v0
; CHECK: br %r14
  %val = call <2 x double> asm "blah $0 $1", "=&v,v" (<2 x double> <double 0.0, double 0.0>)
  ret <2 x double> %val
}

define <4 x float> @f9() {
; CHECK-LABEL: f9:
; CHECK: vgbm  %v0, 0
; CHECK: blah %v24 %v0
; CHECK: br %r14
  %val = call <4 x float> asm "blah $0 $1", "=&v,v" (<4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>)
  ret <4 x float> %val
}

define float @f10() {
; CHECK-LABEL: f10:
; CHECK: lzer %f4
; CHECK: blah %f4
; CHECK: ldr %f0, %f4
; CHECK: br %r14
  %ret = call float asm "blah $0", "={v4},0" (float 0.0)
  ret float %ret
}

define double @f11() {
; CHECK-LABEL: f11:
; CHECK: lzdr %f4
; CHECK: blah %f4
; CHECK: ldr %f0, %f4
; CHECK: br %r14
  %ret = call double asm "blah $0", "={v4},0" (double 0.0)
  ret double %ret
}

define fp128 @f12() {
; CHECK-LABEL: f12:
; CHECK-Z14: vzero %v4
; CHECK: blah %v4
; CHECK: vst %v4, 0(%r2)
; CHECK: br %r14
  %ret = call fp128 asm "blah $0", "={v4},0" (fp128 0xL00000000000000000000000000000000)
  ret fp128 %ret
}

define <2 x i64> @f13() {
; CHECK-LABEL: f13:
; CHECK: vrepig %v4, 1
; CHECK: blah %v4
; CHECK: vlr %v24, %v4
; CHECK: br %r14
  %ret = call <2 x i64> asm "blah $0", "={v4},0" (<2 x i64> <i64 1, i64 1>)
  ret <2 x i64> %ret
}

define <2 x i64> @f14(<2 x i64> %in) {
; CHECK-LABEL: f14:
; CHECK: vlr [[REG:%v[0-9]+]], %v24
; CHECK: blah
; CHECK: vlr %v24, [[REG]]
; CHECK: br %r14
  call void asm sideeffect "blah", "~{v24},~{cc}"()
  ret <2 x i64> %in
}

