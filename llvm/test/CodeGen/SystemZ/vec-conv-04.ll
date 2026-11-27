; Test scalar conversions between integer and float vector elements on z15
; (WCEFB, WCELFB, WCFEB, WCLFEB).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s --check-prefix=Z14

; Test conversion of extracted signed i32 element 0 to f32 (WCEFB).
define float @test_wcefb_0(<4 x i32> %v) {
; CHECK-LABEL: test_wcefb_0:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: wcefb %f0, %v24, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 0
  %f = sitofp i32 %e to float
  ret float %f
}

; Test conversion of extracted signed i32 element 1 to f32 (WCEFB).
define float @test_wcefb_1(<4 x i32> %v) {
; CHECK-LABEL: test_wcefb_1:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 1
; CHECK-NEXT: wcefb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 1
  %f = sitofp i32 %e to float
  ret float %f
}

; Test conversion of extracted signed i32 element 2 to f32 (WCEFB).
define float @test_wcefb_2(<4 x i32> %v) {
; CHECK-LABEL: test_wcefb_2:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 2
; CHECK-NEXT: wcefb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 2
  %f = sitofp i32 %e to float
  ret float %f
}

; Test conversion of extracted signed i32 element 3 to f32 (WCEFB).
define float @test_wcefb_3(<4 x i32> %v) {
; CHECK-LABEL: test_wcefb_3:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 3
; CHECK-NEXT: wcefb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 3
  %f = sitofp i32 %e to float
  ret float %f
}

; Test conversion of extracted unsigned i32 element 0 to f32 (WCELFB).
define float @test_wcelfb_0(<4 x i32> %v) {
; CHECK-LABEL: test_wcelfb_0:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: wcelfb %f0, %v24, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 0
  %f = uitofp i32 %e to float
  ret float %f
}

; Test conversion of extracted unsigned i32 element 1 to f32 (WCELFB).
define float @test_wcelfb_1(<4 x i32> %v) {
; CHECK-LABEL: test_wcelfb_1:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 1
; CHECK-NEXT: wcelfb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 1
  %f = uitofp i32 %e to float
  ret float %f
}

; Test conversion of extracted unsigned i32 element 2 to f32 (WCELFB).
define float @test_wcelfb_2(<4 x i32> %v) {
; CHECK-LABEL: test_wcelfb_2:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 2
; CHECK-NEXT: wcelfb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 2
  %f = uitofp i32 %e to float
  ret float %f
}

; Test conversion of extracted unsigned i32 element 3 to f32 (WCELFB).
define float @test_wcelfb_3(<4 x i32> %v) {
; CHECK-LABEL: test_wcelfb_3:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 3
; CHECK-NEXT: wcelfb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 3
  %f = uitofp i32 %e to float
  ret float %f
}

; Test conversion of scalar f32 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wcfeb_scalar(float %f) {
; CHECK-LABEL: test_wcfeb_scalar:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wcfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %i = fptosi float %f to i32
  %v = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %v
}

; Test conversion of extracted f32 element 0 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wcfeb_vec_0(<4 x float> %v) {
; CHECK-LABEL: test_wcfeb_vec_0:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wcfeb %v24, %v24, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 0
  %i = fptosi float %e to i32
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 1 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wcfeb_vec_1(<4 x float> %v) {
; CHECK-LABEL: test_wcfeb_vec_1:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 1
; CHECK-NEXT: wcfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 1
  %i = fptosi float %e to i32
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 2 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wcfeb_vec_2(<4 x float> %v) {
; CHECK-LABEL: test_wcfeb_vec_2:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 2
; CHECK-NEXT: wcfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 2
  %i = fptosi float %e to i32
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 3 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wcfeb_vec_3(<4 x float> %v) {
; CHECK-LABEL: test_wcfeb_vec_3:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 3
; CHECK-NEXT: wcfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 3
  %i = fptosi float %e to i32
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of scalar f32 to signed i32 splatted to all vector elements (WCFEB).
define <4 x i32> @test_wcfeb_splat(float %f) {
; CHECK-LABEL: test_wcfeb_splat:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wcfeb %f0, %f0, 0, 5
; CHECK-NEXT: vrepf %v24, %v0, 0
; CHECK-NEXT: br %r14
  %i = fptosi float %f to i32
  %v = insertelement <4 x i32> poison, i32 %i, i32 0
  %splat = shufflevector <4 x i32> %v, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %splat
}

; Test conversion of scalar f32 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wclfeb_scalar(float %f) {
; CHECK-LABEL: test_wclfeb_scalar:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wclfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %i = fptoui float %f to i32
  %v = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %v
}

; Test conversion of extracted f32 element 0 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wclfeb_vec_0(<4 x float> %v) {
; CHECK-LABEL: test_wclfeb_vec_0:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wclfeb %v24, %v24, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 0
  %i = fptoui float %e to i32
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 1 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wclfeb_vec_1(<4 x float> %v) {
; CHECK-LABEL: test_wclfeb_vec_1:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 1
; CHECK-NEXT: wclfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 1
  %i = fptoui float %e to i32
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 2 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wclfeb_vec_2(<4 x float> %v) {
; CHECK-LABEL: test_wclfeb_vec_2:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 2
; CHECK-NEXT: wclfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 2
  %i = fptoui float %e to i32
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 3 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_wclfeb_vec_3(<4 x float> %v) {
; CHECK-LABEL: test_wclfeb_vec_3:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 3
; CHECK-NEXT: wclfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 3
  %i = fptoui float %e to i32
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of scalar f32 to unsigned i32 splatted to all vector elements (WCLFEB).
define <4 x i32> @test_wclfeb_splat(float %f) {
; CHECK-LABEL: test_wclfeb_splat:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wclfeb %f0, %f0, 0, 5
; CHECK-NEXT: vrepf %v24, %v0, 0
; CHECK-NEXT: br %r14
  %i = fptoui float %f to i32
  %v = insertelement <4 x i32> poison, i32 %i, i32 0
  %splat = shufflevector <4 x i32> %v, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %splat
}

; The following conversions have no vector operand and must not use the
; vector forms.

; Test conversion of a GPR value to f32.
define float @test_gpr_to_f32(i32 %x) {
; CHECK-LABEL: test_gpr_to_f32:
; CHECK-NOT: wcefb
; CHECK: cefbr %f0, %r2
; CHECK-NEXT: br %r14
  %f = sitofp i32 %x to float
  ret float %f
}

; Test conversion of a GPR value to f32 (unsigned).
define float @test_gpr_to_f32_unsigned(i32 %x) {
; CHECK-LABEL: test_gpr_to_f32_unsigned:
; CHECK-NOT: wcelfb
; CHECK: celfbr %f0, 0, %r2, 0
; CHECK-NEXT: br %r14
  %f = uitofp i32 %x to float
  ret float %f
}

; Test conversion of f32 to a GPR value.
define i32 @test_f32_to_gpr(float %x) {
; CHECK-LABEL: test_f32_to_gpr:
; CHECK-NOT: wcfeb
; CHECK: cfebr %r2, 5, %f0
; CHECK-NEXT: br %r14
  %i = fptosi float %x to i32
  ret i32 %i
}

; Test conversion of f32 to a GPR value (unsigned).
define i32 @test_f32_to_gpr_unsigned(float %x) {
; CHECK-LABEL: test_f32_to_gpr_unsigned:
; CHECK-NOT: wclfeb
; CHECK: clfebr %r2, 5, %f0, 0
; CHECK-NEXT: br %r14
  %i = fptoui float %x to i32
  ret i32 %i
}

; Test conversion of a variably-indexed i32 element to f32.
define float @test_var_index(<4 x i32> %v, i32 %i) {
; CHECK-LABEL: test_var_index:
; CHECK-NOT: wcelfb
; CHECK: vlgvf %r0, %v24, 0(%r2)
; CHECK-NEXT: celfbr %f0, 0, %r0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 %i
  %f = uitofp i32 %e to float
  ret float %f
}

; Test conversion of f32 inserted into a defined vector.  The other elements
; must be preserved, so WCLFEB cannot be used.
define <4 x i32> @test_insert_defined(<4 x i32> %v, float %f) {
; CHECK-LABEL: test_insert_defined:
; CHECK-NOT: wclfeb
; CHECK: clfebr %r0, 5, %f0, 0
; CHECK-NEXT: vlvgf %v24, %r0, 0
; CHECK-NEXT: br %r14
  %i = fptoui float %f to i32
  %r = insertelement <4 x i32> %v, i32 %i, i32 0
  ret <4 x i32> %r
}

; Without the z15 vector enhancements the scalar conversions must go through
; GPRs.

; Z14-LABEL: test_wcefb_0:
; Z14-NOT: wcefb %
; Z14: vlgvf %r0, %v24, 0
; Z14-NEXT: cefbr %f0, %r0

; Z14-LABEL: test_wcelfb_1:
; Z14-NOT: wcelfb %
; Z14: vlgvf %r0, %v24, 1
; Z14-NEXT: celfbr %f0, 0, %r0, 0

; Z14-LABEL: test_wcfeb_scalar:
; Z14-NOT: wcfeb %
; Z14: cfebr %r0, 5, %f0
; Z14-NEXT: vlvg

; Z14-LABEL: test_wclfeb_splat:
; Z14-NOT: wclfeb %
; Z14: clfebr %r0, 5, %f0, 0
; Z14-NEXT: vlvgp %v0, %r0, %r0
; Z14-NEXT: vrepf %v24, %v0, 1
