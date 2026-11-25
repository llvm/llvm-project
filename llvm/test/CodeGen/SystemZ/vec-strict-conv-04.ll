; Test strict scalar conversions between integer and float vector elements on
; z15 (WCEFB, WCELFB, WCFEB, WCLFEB).
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z15 | FileCheck %s
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s --check-prefix=Z14

declare float @llvm.experimental.constrained.sitofp.f32.i32(i32, metadata, metadata)
declare float @llvm.experimental.constrained.uitofp.f32.i32(i32, metadata, metadata)
declare i32 @llvm.experimental.constrained.fptosi.i32.f32(float, metadata)
declare i32 @llvm.experimental.constrained.fptoui.i32.f32(float, metadata)

; Test conversion of extracted signed i32 element 0 to f32 (WCEFB).
define float @test_strict_wcefb_0(<4 x i32> %v) #0 {
; CHECK-LABEL: test_strict_wcefb_0:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: wcefb %f0, %v24, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 0
  %f = call float @llvm.experimental.constrained.sitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of extracted signed i32 element 1 to f32 (WCEFB).
define float @test_strict_wcefb_1(<4 x i32> %v) #0 {
; CHECK-LABEL: test_strict_wcefb_1:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 1
; CHECK-NEXT: wcefb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 1
  %f = call float @llvm.experimental.constrained.sitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of extracted signed i32 element 2 to f32 (WCEFB).
define float @test_strict_wcefb_2(<4 x i32> %v) #0 {
; CHECK-LABEL: test_strict_wcefb_2:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 2
; CHECK-NEXT: wcefb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 2
  %f = call float @llvm.experimental.constrained.sitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of extracted signed i32 element 3 to f32 (WCEFB).
define float @test_strict_wcefb_3(<4 x i32> %v) #0 {
; CHECK-LABEL: test_strict_wcefb_3:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 3
; CHECK-NEXT: wcefb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 3
  %f = call float @llvm.experimental.constrained.sitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of extracted unsigned i32 element 0 to f32 (WCELFB).
define float @test_strict_wcelfb_0(<4 x i32> %v) #0 {
; CHECK-LABEL: test_strict_wcelfb_0:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: wcelfb %f0, %v24, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 0
  %f = call float @llvm.experimental.constrained.uitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of extracted unsigned i32 element 1 to f32 (WCELFB).
define float @test_strict_wcelfb_1(<4 x i32> %v) #0 {
; CHECK-LABEL: test_strict_wcelfb_1:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 1
; CHECK-NEXT: wcelfb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 1
  %f = call float @llvm.experimental.constrained.uitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of extracted unsigned i32 element 2 to f32 (WCELFB).
define float @test_strict_wcelfb_2(<4 x i32> %v) #0 {
; CHECK-LABEL: test_strict_wcelfb_2:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 2
; CHECK-NEXT: wcelfb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 2
  %f = call float @llvm.experimental.constrained.uitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of extracted unsigned i32 element 3 to f32 (WCELFB).
define float @test_strict_wcelfb_3(<4 x i32> %v) #0 {
; CHECK-LABEL: test_strict_wcelfb_3:
; CHECK-NOT: {{vlgvf|cefbr|celfbr}}
; CHECK: vrepf %v0, %v24, 3
; CHECK-NEXT: wcelfb %f0, %f0, 0, 0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 3
  %f = call float @llvm.experimental.constrained.uitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of scalar f32 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wcfeb_scalar(float %f) #0 {
; CHECK-LABEL: test_strict_wcfeb_scalar:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wcfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %i = call i32 @llvm.experimental.constrained.fptosi.i32.f32(
      float %f, metadata !"fpexcept.strict") #0
  %v = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %v
}

; Test conversion of extracted f32 element 0 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wcfeb_vec_0(<4 x float> %v) #0 {
; CHECK-LABEL: test_strict_wcfeb_vec_0:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wcfeb %v24, %v24, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 0
  %i = call i32 @llvm.experimental.constrained.fptosi.i32.f32(
      float %e, metadata !"fpexcept.strict") #0
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 1 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wcfeb_vec_1(<4 x float> %v) #0 {
; CHECK-LABEL: test_strict_wcfeb_vec_1:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 1
; CHECK-NEXT: wcfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 1
  %i = call i32 @llvm.experimental.constrained.fptosi.i32.f32(
      float %e, metadata !"fpexcept.strict") #0
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 2 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wcfeb_vec_2(<4 x float> %v) #0 {
; CHECK-LABEL: test_strict_wcfeb_vec_2:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 2
; CHECK-NEXT: wcfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 2
  %i = call i32 @llvm.experimental.constrained.fptosi.i32.f32(
      float %e, metadata !"fpexcept.strict") #0
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 3 to signed i32 in vector element 0 (WCFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wcfeb_vec_3(<4 x float> %v) #0 {
; CHECK-LABEL: test_strict_wcfeb_vec_3:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 3
; CHECK-NEXT: wcfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 3
  %i = call i32 @llvm.experimental.constrained.fptosi.i32.f32(
      float %e, metadata !"fpexcept.strict") #0
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of scalar f32 to signed i32 splatted to all vector elements (WCFEB).
define <4 x i32> @test_strict_wcfeb_splat(float %f) #0 {
; CHECK-LABEL: test_strict_wcfeb_splat:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wcfeb %f0, %f0, 0, 5
; CHECK-NEXT: vrepf %v24, %v0, 0
; CHECK-NEXT: br %r14
  %i = call i32 @llvm.experimental.constrained.fptosi.i32.f32(
      float %f, metadata !"fpexcept.strict") #0
  %v = insertelement <4 x i32> poison, i32 %i, i32 0
  %splat = shufflevector <4 x i32> %v, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %splat
}

; Test conversion of scalar f32 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wclfeb_scalar(float %f) #0 {
; CHECK-LABEL: test_strict_wclfeb_scalar:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wclfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %i = call i32 @llvm.experimental.constrained.fptoui.i32.f32(
      float %f, metadata !"fpexcept.strict") #0
  %v = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %v
}

; Test conversion of extracted f32 element 0 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wclfeb_vec_0(<4 x float> %v) #0 {
; CHECK-LABEL: test_strict_wclfeb_vec_0:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wclfeb %v24, %v24, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 0
  %i = call i32 @llvm.experimental.constrained.fptoui.i32.f32(
      float %e, metadata !"fpexcept.strict") #0
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 1 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wclfeb_vec_1(<4 x float> %v) #0 {
; CHECK-LABEL: test_strict_wclfeb_vec_1:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 1
; CHECK-NEXT: wclfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 1
  %i = call i32 @llvm.experimental.constrained.fptoui.i32.f32(
      float %e, metadata !"fpexcept.strict") #0
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 2 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wclfeb_vec_2(<4 x float> %v) #0 {
; CHECK-LABEL: test_strict_wclfeb_vec_2:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 2
; CHECK-NEXT: wclfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 2
  %i = call i32 @llvm.experimental.constrained.fptoui.i32.f32(
      float %e, metadata !"fpexcept.strict") #0
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of extracted f32 element 3 to unsigned i32 in vector element 0 (WCLFEB).
; The other vector elements are undefined.
define <4 x i32> @test_strict_wclfeb_vec_3(<4 x float> %v) #0 {
; CHECK-LABEL: test_strict_wclfeb_vec_3:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: vrepf %v0, %v24, 3
; CHECK-NEXT: wclfeb %v24, %f0, 0, 5
; CHECK-NEXT: br %r14
  %e = extractelement <4 x float> %v, i32 3
  %i = call i32 @llvm.experimental.constrained.fptoui.i32.f32(
      float %e, metadata !"fpexcept.strict") #0
  %res = insertelement <4 x i32> poison, i32 %i, i32 0
  ret <4 x i32> %res
}

; Test conversion of scalar f32 to unsigned i32 splatted to all vector elements (WCLFEB).
define <4 x i32> @test_strict_wclfeb_splat(float %f) #0 {
; CHECK-LABEL: test_strict_wclfeb_splat:
; CHECK-NOT: {{cfebr|clfebr|vlvg}}
; CHECK: wclfeb %f0, %f0, 0, 5
; CHECK-NEXT: vrepf %v24, %v0, 0
; CHECK-NEXT: br %r14
  %i = call i32 @llvm.experimental.constrained.fptoui.i32.f32(
      float %f, metadata !"fpexcept.strict") #0
  %v = insertelement <4 x i32> poison, i32 %i, i32 0
  %splat = shufflevector <4 x i32> %v, <4 x i32> poison, <4 x i32> zeroinitializer
  ret <4 x i32> %splat
}

; The following conversions have no vector operand and must not use the
; vector forms.

; Test conversion of a GPR value to f32.
define float @test_strict_gpr_to_f32(i32 %x) #0 {
; CHECK-LABEL: test_strict_gpr_to_f32:
; CHECK-NOT: wcefb
; CHECK: cefbr %f0, %r2
; CHECK-NEXT: br %r14
  %f = call float @llvm.experimental.constrained.sitofp.f32.i32(
      i32 %x, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Test conversion of f32 to a GPR value.
define i32 @test_strict_f32_to_gpr(float %x) #0 {
; CHECK-LABEL: test_strict_f32_to_gpr:
; CHECK-NOT: wclfeb
; CHECK: clfebr %r2, 5, %f0, 0
; CHECK-NEXT: br %r14
  %i = call i32 @llvm.experimental.constrained.fptoui.i32.f32(
      float %x, metadata !"fpexcept.strict") #0
  ret i32 %i
}

; Test conversion of a variably-indexed i32 element to f32.
define float @test_strict_var_index(<4 x i32> %v, i32 %i) #0 {
; CHECK-LABEL: test_strict_var_index:
; CHECK-NOT: wcefb
; CHECK: vlgvf %r0, %v24, 0(%r2)
; CHECK-NEXT: cefbr %f0, %r0
; CHECK-NEXT: br %r14
  %e = extractelement <4 x i32> %v, i32 %i
  %f = call float @llvm.experimental.constrained.sitofp.f32.i32(
      i32 %e, metadata !"round.dynamic", metadata !"fpexcept.strict") #0
  ret float %f
}

; Without the z15 vector enhancements the scalar conversions must go through
; GPRs.

; Z14-LABEL: test_strict_wcefb_0:
; Z14-NOT: wcefb %
; Z14: vlgvf %r0, %v24, 0
; Z14-NEXT: cefbr %f0, %r0

; Z14-LABEL: test_strict_wcelfb_1:
; Z14-NOT: wcelfb %
; Z14: vlgvf %r0, %v24, 1
; Z14-NEXT: celfbr %f0, 0, %r0, 0

; Z14-LABEL: test_strict_wcfeb_scalar:
; Z14-NOT: wcfeb %
; Z14: cfebr %r0, 5, %f0
; Z14-NEXT: vlvg

; Z14-LABEL: test_strict_wclfeb_splat:
; Z14-NOT: wclfeb %
; Z14: clfebr %r0, 5, %f0, 0
; Z14-NEXT: vlvgp %v0, %r0, %r0
; Z14-NEXT: vrepf %v24, %v0, 1

attributes #0 = { strictfp }
