; RUN: llc -mtriple=aarch64-none-linux-gnu -mattr=+neon < %s | FileCheck %s

;; Scalar Floating-point Compare

define i32 @test_vceqs_f32(float %a, float %b) {
; CHECK: test_vceqs_f32
; CHECK: fcmeq {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vceq.i = insertelement <1 x float> undef, float %a, i32 0
  %vceq1.i = insertelement <1 x float> undef, float %b, i32 0
  %vceq2.i = call <1 x i32> @llvm.aarch64.neon.vceq.v1i32.v1f32.v1f32(<1 x float> %vceq.i, <1 x float> %vceq1.i)
  %0 = extractelement <1 x i32> %vceq2.i, i32 0
  ret i32 %0
}

define i64 @test_vceqd_f64(double %a, double %b) {
; CHECK: test_vceqd_f64
; CHECK: fcmeq {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vceq.i = insertelement <1 x double> undef, double %a, i32 0
  %vceq1.i = insertelement <1 x double> undef, double %b, i32 0
  %vceq2.i = call <1 x i64> @llvm.aarch64.neon.vceq.v1i64.v1f64.v1f64(<1 x double> %vceq.i, <1 x double> %vceq1.i)
  %0 = extractelement <1 x i64> %vceq2.i, i32 0
  ret i64 %0
}

define <1 x i64> @test_vceqz_f64(<1 x double> %a) #0 {
; CHECK: test_vceqz_f64
; CHECK: fcmeq  {{d[0-9]+}}, {{d[0-9]+}}, #0.0
entry:
  %0 = fcmp oeq <1 x double> %a, zeroinitializer
  %vceqz.i = zext <1 x i1> %0 to <1 x i64>
  ret <1 x i64> %vceqz.i
}

define i32 @test_vceqzs_f32(float %a) {
; CHECK: test_vceqzs_f32
; CHECK: fcmeq {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %vceq.i = insertelement <1 x float> undef, float %a, i32 0
  %vceq1.i = call <1 x i32> @llvm.aarch64.neon.vceq.v1i32.v1f32.v1f32(<1 x float> %vceq.i, <1 x float> zeroinitializer)
  %0 = extractelement <1 x i32> %vceq1.i, i32 0
  ret i32 %0
}

define i64 @test_vceqzd_f64(double %a) {
; CHECK: test_vceqzd_f64
; CHECK: fcmeq {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %vceq.i = insertelement <1 x double> undef, double %a, i32 0
  %vceq1.i = tail call <1 x i64> @llvm.aarch64.neon.vceq.v1i64.v1f64.v1f32(<1 x double> %vceq.i, <1 x float> zeroinitializer) #5
  %0 = extractelement <1 x i64> %vceq1.i, i32 0
  ret i64 %0
}

define i32 @test_vcges_f32(float %a, float %b) {
; CHECK: test_vcges_f32
; CHECK: fcmge {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcge.i = insertelement <1 x float> undef, float %a, i32 0
  %vcge1.i = insertelement <1 x float> undef, float %b, i32 0
  %vcge2.i = call <1 x i32> @llvm.aarch64.neon.vcge.v1i32.v1f32.v1f32(<1 x float> %vcge.i, <1 x float> %vcge1.i)
  %0 = extractelement <1 x i32> %vcge2.i, i32 0
  ret i32 %0
}

define i64 @test_vcged_f64(double %a, double %b) {
; CHECK: test_vcged_f64
; CHECK: fcmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcge.i = insertelement <1 x double> undef, double %a, i32 0
  %vcge1.i = insertelement <1 x double> undef, double %b, i32 0
  %vcge2.i = call <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1f64.v1f64(<1 x double> %vcge.i, <1 x double> %vcge1.i)
  %0 = extractelement <1 x i64> %vcge2.i, i32 0
  ret i64 %0
}

define i32 @test_vcgezs_f32(float %a) {
; CHECK: test_vcgezs_f32
; CHECK: fcmge {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %vcge.i = insertelement <1 x float> undef, float %a, i32 0
  %vcge1.i = call <1 x i32> @llvm.aarch64.neon.vcge.v1i32.v1f32.v1f32(<1 x float> %vcge.i, <1 x float> zeroinitializer)
  %0 = extractelement <1 x i32> %vcge1.i, i32 0
  ret i32 %0
}

define i64 @test_vcgezd_f64(double %a) {
; CHECK: test_vcgezd_f64
; CHECK: fcmge {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %vcge.i = insertelement <1 x double> undef, double %a, i32 0
  %vcge1.i = tail call <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1f64.v1f32(<1 x double> %vcge.i, <1 x float> zeroinitializer) #5
  %0 = extractelement <1 x i64> %vcge1.i, i32 0
  ret i64 %0
}

define i32 @test_vcgts_f32(float %a, float %b) {
; CHECK: test_vcgts_f32
; CHECK: fcmgt {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcgt.i = insertelement <1 x float> undef, float %a, i32 0
  %vcgt1.i = insertelement <1 x float> undef, float %b, i32 0
  %vcgt2.i = call <1 x i32> @llvm.aarch64.neon.vcgt.v1i32.v1f32.v1f32(<1 x float> %vcgt.i, <1 x float> %vcgt1.i)
  %0 = extractelement <1 x i32> %vcgt2.i, i32 0
  ret i32 %0
}

define i64 @test_vcgtd_f64(double %a, double %b) {
; CHECK: test_vcgtd_f64
; CHECK: fcmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcgt.i = insertelement <1 x double> undef, double %a, i32 0
  %vcgt1.i = insertelement <1 x double> undef, double %b, i32 0
  %vcgt2.i = call <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1f64.v1f64(<1 x double> %vcgt.i, <1 x double> %vcgt1.i)
  %0 = extractelement <1 x i64> %vcgt2.i, i32 0
  ret i64 %0
}

define i32 @test_vcgtzs_f32(float %a) {
; CHECK: test_vcgtzs_f32
; CHECK: fcmgt {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %vcgt.i = insertelement <1 x float> undef, float %a, i32 0
  %vcgt1.i = call <1 x i32> @llvm.aarch64.neon.vcgt.v1i32.v1f32.v1f32(<1 x float> %vcgt.i, <1 x float> zeroinitializer)
  %0 = extractelement <1 x i32> %vcgt1.i, i32 0
  ret i32 %0
}

define i64 @test_vcgtzd_f64(double %a) {
; CHECK: test_vcgtzd_f64
; CHECK: fcmgt {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %vcgt.i = insertelement <1 x double> undef, double %a, i32 0
  %vcgt1.i = tail call <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1f64.v1f32(<1 x double> %vcgt.i, <1 x float> zeroinitializer) #5
  %0 = extractelement <1 x i64> %vcgt1.i, i32 0
  ret i64 %0
}

define i32 @test_vcles_f32(float %a, float %b) {
; CHECK: test_vcles_f32
; CHECK: fcmge {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcge.i = insertelement <1 x float> undef, float %a, i32 0
  %vcge1.i = insertelement <1 x float> undef, float %b, i32 0
  %vcge2.i = call <1 x i32> @llvm.aarch64.neon.vcge.v1i32.v1f32.v1f32(<1 x float> %vcge.i, <1 x float> %vcge1.i)
  %0 = extractelement <1 x i32> %vcge2.i, i32 0
  ret i32 %0
}

define i64 @test_vcled_f64(double %a, double %b) {
; CHECK: test_vcled_f64
; CHECK: fcmge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcge.i = insertelement <1 x double> undef, double %a, i32 0
  %vcge1.i = insertelement <1 x double> undef, double %b, i32 0
  %vcge2.i = call <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1f64.v1f64(<1 x double> %vcge.i, <1 x double> %vcge1.i)
  %0 = extractelement <1 x i64> %vcge2.i, i32 0
  ret i64 %0
}

define i32 @test_vclezs_f32(float %a) {
; CHECK: test_vclezs_f32
; CHECK: fcmle {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %vcle.i = insertelement <1 x float> undef, float %a, i32 0
  %vcle1.i = call <1 x i32> @llvm.aarch64.neon.vclez.v1i32.v1f32.v1f32(<1 x float> %vcle.i, <1 x float> zeroinitializer)
  %0 = extractelement <1 x i32> %vcle1.i, i32 0
  ret i32 %0
}

define i64 @test_vclezd_f64(double %a) {
; CHECK: test_vclezd_f64
; CHECK: fcmle {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %vcle.i = insertelement <1 x double> undef, double %a, i32 0
  %vcle1.i = tail call <1 x i64> @llvm.aarch64.neon.vclez.v1i64.v1f64.v1f32(<1 x double> %vcle.i, <1 x float> zeroinitializer) #5
  %0 = extractelement <1 x i64> %vcle1.i, i32 0
  ret i64 %0
}

define i32 @test_vclts_f32(float %a, float %b) {
; CHECK: test_vclts_f32
; CHECK: fcmgt {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcgt.i = insertelement <1 x float> undef, float %b, i32 0
  %vcgt1.i = insertelement <1 x float> undef, float %a, i32 0
  %vcgt2.i = call <1 x i32> @llvm.aarch64.neon.vcgt.v1i32.v1f32.v1f32(<1 x float> %vcgt.i, <1 x float> %vcgt1.i)
  %0 = extractelement <1 x i32> %vcgt2.i, i32 0
  ret i32 %0
}

define i64 @test_vcltd_f64(double %a, double %b) {
; CHECK: test_vcltd_f64
; CHECK: fcmgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcgt.i = insertelement <1 x double> undef, double %b, i32 0
  %vcgt1.i = insertelement <1 x double> undef, double %a, i32 0
  %vcgt2.i = call <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1f64.v1f64(<1 x double> %vcgt.i, <1 x double> %vcgt1.i)
  %0 = extractelement <1 x i64> %vcgt2.i, i32 0
  ret i64 %0
}

define i32 @test_vcltzs_f32(float %a) {
; CHECK: test_vcltzs_f32
; CHECK: fcmlt {{s[0-9]}}, {{s[0-9]}}, #0.0
entry:
  %vclt.i = insertelement <1 x float> undef, float %a, i32 0
  %vclt1.i = call <1 x i32> @llvm.aarch64.neon.vcltz.v1i32.v1f32.v1f32(<1 x float> %vclt.i, <1 x float> zeroinitializer)
  %0 = extractelement <1 x i32> %vclt1.i, i32 0
  ret i32 %0
}

define i64 @test_vcltzd_f64(double %a) {
; CHECK: test_vcltzd_f64
; CHECK: fcmlt {{d[0-9]}}, {{d[0-9]}}, #0.0
entry:
  %vclt.i = insertelement <1 x double> undef, double %a, i32 0
  %vclt1.i = tail call <1 x i64> @llvm.aarch64.neon.vcltz.v1i64.v1f64.v1f32(<1 x double> %vclt.i, <1 x float> zeroinitializer) #5
  %0 = extractelement <1 x i64> %vclt1.i, i32 0
  ret i64 %0
}

define i32 @test_vcages_f32(float %a, float %b) {
; CHECK: test_vcages_f32
; CHECK: facge {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcage.i = insertelement <1 x float> undef, float %a, i32 0
  %vcage1.i = insertelement <1 x float> undef, float %b, i32 0
  %vcage2.i = call <1 x i32> @llvm.aarch64.neon.vcage.v1i32.v1f32.v1f32(<1 x float> %vcage.i, <1 x float> %vcage1.i)
  %0 = extractelement <1 x i32> %vcage2.i, i32 0
  ret i32 %0
}

define i64 @test_vcaged_f64(double %a, double %b) {
; CHECK: test_vcaged_f64
; CHECK: facge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcage.i = insertelement <1 x double> undef, double %a, i32 0
  %vcage1.i = insertelement <1 x double> undef, double %b, i32 0
  %vcage2.i = call <1 x i64> @llvm.aarch64.neon.vcage.v1i64.v1f64.v1f64(<1 x double> %vcage.i, <1 x double> %vcage1.i)
  %0 = extractelement <1 x i64> %vcage2.i, i32 0
  ret i64 %0
}

define i32 @test_vcagts_f32(float %a, float %b) {
; CHECK: test_vcagts_f32
; CHECK: facgt {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcagt.i = insertelement <1 x float> undef, float %a, i32 0
  %vcagt1.i = insertelement <1 x float> undef, float %b, i32 0
  %vcagt2.i = call <1 x i32> @llvm.aarch64.neon.vcagt.v1i32.v1f32.v1f32(<1 x float> %vcagt.i, <1 x float> %vcagt1.i)
  %0 = extractelement <1 x i32> %vcagt2.i, i32 0
  ret i32 %0
}

define i64 @test_vcagtd_f64(double %a, double %b) {
; CHECK: test_vcagtd_f64
; CHECK: facgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcagt.i = insertelement <1 x double> undef, double %a, i32 0
  %vcagt1.i = insertelement <1 x double> undef, double %b, i32 0
  %vcagt2.i = call <1 x i64> @llvm.aarch64.neon.vcagt.v1i64.v1f64.v1f64(<1 x double> %vcagt.i, <1 x double> %vcagt1.i)
  %0 = extractelement <1 x i64> %vcagt2.i, i32 0
  ret i64 %0
}

define i32 @test_vcales_f32(float %a, float %b) {
; CHECK: test_vcales_f32
; CHECK: facge {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcage.i = insertelement <1 x float> undef, float %b, i32 0
  %vcage1.i = insertelement <1 x float> undef, float %a, i32 0
  %vcage2.i = call <1 x i32> @llvm.aarch64.neon.vcage.v1i32.v1f32.v1f32(<1 x float> %vcage.i, <1 x float> %vcage1.i)
  %0 = extractelement <1 x i32> %vcage2.i, i32 0
  ret i32 %0
}

define i64 @test_vcaled_f64(double %a, double %b) {
; CHECK: test_vcaled_f64
; CHECK: facge {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcage.i = insertelement <1 x double> undef, double %b, i32 0
  %vcage1.i = insertelement <1 x double> undef, double %a, i32 0
  %vcage2.i = call <1 x i64> @llvm.aarch64.neon.vcage.v1i64.v1f64.v1f64(<1 x double> %vcage.i, <1 x double> %vcage1.i)
  %0 = extractelement <1 x i64> %vcage2.i, i32 0
  ret i64 %0
}

define i32 @test_vcalts_f32(float %a, float %b) {
; CHECK: test_vcalts_f32
; CHECK: facgt {{s[0-9]}}, {{s[0-9]}}, {{s[0-9]}}
entry:
  %vcalt.i = insertelement <1 x float> undef, float %b, i32 0
  %vcalt1.i = insertelement <1 x float> undef, float %a, i32 0
  %vcalt2.i = call <1 x i32> @llvm.aarch64.neon.vcagt.v1i32.v1f32.v1f32(<1 x float> %vcalt.i, <1 x float> %vcalt1.i)
  %0 = extractelement <1 x i32> %vcalt2.i, i32 0
  ret i32 %0
}

define i64 @test_vcaltd_f64(double %a, double %b) {
; CHECK: test_vcaltd_f64
; CHECK: facgt {{d[0-9]}}, {{d[0-9]}}, {{d[0-9]}}
entry:
  %vcalt.i = insertelement <1 x double> undef, double %b, i32 0
  %vcalt1.i = insertelement <1 x double> undef, double %a, i32 0
  %vcalt2.i = call <1 x i64> @llvm.aarch64.neon.vcagt.v1i64.v1f64.v1f64(<1 x double> %vcalt.i, <1 x double> %vcalt1.i)
  %0 = extractelement <1 x i64> %vcalt2.i, i32 0
  ret i64 %0
}

declare <1 x i32> @llvm.aarch64.neon.vceq.v1i32.v1f32.v1f32(<1 x float>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vceq.v1i64.v1f64.v1f32(<1 x double>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vceq.v1i64.v1f64.v1f64(<1 x double>, <1 x double>)
declare <1 x i32> @llvm.aarch64.neon.vcge.v1i32.v1f32.v1f32(<1 x float>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1f64.v1f32(<1 x double>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vcge.v1i64.v1f64.v1f64(<1 x double>, <1 x double>)
declare <1 x i32> @llvm.aarch64.neon.vclez.v1i32.v1f32.v1f32(<1 x float>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vclez.v1i64.v1f64.v1f32(<1 x double>, <1 x float>)
declare <1 x i32> @llvm.aarch64.neon.vcgt.v1i32.v1f32.v1f32(<1 x float>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1f64.v1f32(<1 x double>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vcgt.v1i64.v1f64.v1f64(<1 x double>, <1 x double>)
declare <1 x i32> @llvm.aarch64.neon.vcltz.v1i32.v1f32.v1f32(<1 x float>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vcltz.v1i64.v1f64.v1f32(<1 x double>, <1 x float>)
declare <1 x i32> @llvm.aarch64.neon.vcage.v1i32.v1f32.v1f32(<1 x float>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vcage.v1i64.v1f64.v1f64(<1 x double>, <1 x double>)
declare <1 x i32> @llvm.aarch64.neon.vcagt.v1i32.v1f32.v1f32(<1 x float>, <1 x float>)
declare <1 x i64> @llvm.aarch64.neon.vcagt.v1i64.v1f64.v1f64(<1 x double>, <1 x double>)
