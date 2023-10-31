; RUN: llc < %s -verify-machineinstrs -mtriple=arm64-none-linux-gnu -mattr=+neon -fp-contract=fast | FileCheck %s
; arm64 has a separate copy as aarch64-neon-v1i1-setcc.ll

; This file test the DAG node like "v1i1 SETCC v1i64, v1i64". As the v1i1 type
; is illegal in AArch64 backend, the legalizer tries to scalarize this node.
; As the v1i64 operands of SETCC are legal types, they will not be scalarized.
; Currently the type legalizer will have an assertion failure as it assumes all
; operands of SETCC have been legalized.
; FIXME: If the algorithm of type scalarization is improved and can legaize
; "v1i1 SETCC" correctly, these test cases are not needed.

define i64 @test_sext_extr_cmp_0(<1 x i64> %v1, <1 x i64> %v2) {
; CHECK-LABEL: test_sext_extr_cmp_0:
; CHECK: cmp {{x[0-9]+}}, {{x[0-9]+}}
  %1 = icmp sge <1 x i64> %v1, %v2
  %2 = extractelement <1 x i1> %1, i32 0
  %vget_lane = sext i1 %2 to i64
  ret i64 %vget_lane
}

define i64 @test_sext_extr_cmp_1(<1 x double> %v1, <1 x double> %v2) {
; CHECK-LABEL: test_sext_extr_cmp_1:
; CHECK: fcmp {{d[0-9]+}}, {{d[0-9]+}}
  %1 = fcmp oeq <1 x double> %v1, %v2
  %2 = extractelement <1 x i1> %1, i32 0
  %vget_lane = sext i1 %2 to i64
  ret i64 %vget_lane
}

define <1 x i64> @test_select_v1i1_0(<1 x i64> %v1, <1 x i64> %v2, <1 x i64> %v3) {
; CHECK-LABEL: test_select_v1i1_0:
; CHECK: cmeq d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: bic v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %1 = icmp eq <1 x i64> %v1, %v2
  %res = select <1 x i1> %1, <1 x i64> zeroinitializer, <1 x i64> %v3
  ret <1 x i64> %res
}

define <1 x i64> @test_select_v1i1_1(<1 x double> %v1, <1 x double> %v2, <1 x i64> %v3) {
; CHECK-LABEL: test_select_v1i1_1:
; CHECK: fcmeq d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: bic v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %1 = fcmp oeq <1 x double> %v1, %v2
  %res = select <1 x i1> %1, <1 x i64> zeroinitializer, <1 x i64> %v3
  ret <1 x i64> %res
}

define <1 x double> @test_select_v1i1_2(<1 x i64> %v1, <1 x i64> %v2, <1 x double> %v3) {
; CHECK-LABEL: test_select_v1i1_2:
; CHECK: cmeq d{{[0-9]+}}, d{{[0-9]+}}, d{{[0-9]+}}
; CHECK: bic v{{[0-9]+}}.8b, v{{[0-9]+}}.8b, v{{[0-9]+}}.8b
  %1 = icmp eq <1 x i64> %v1, %v2
  %res = select <1 x i1> %1, <1 x double> zeroinitializer, <1 x double> %v3
  ret <1 x double> %res
}

define i32 @test_br_extr_cmp(<1 x i64> %v1, <1 x i64> %v2) {
; CHECK-LABEL: test_br_extr_cmp:
; CHECK: cmp x{{[0-9]+}}, x{{[0-9]+}}
  %1 = icmp eq <1 x i64> %v1, %v2
  %2 = extractelement <1 x i1> %1, i32 0
  br i1 %2, label %if.end, label %if.then

if.then:
  ret i32 0;

if.end:
  ret i32 1;
}

define dso_local <1 x half> @cmp_select(<1 x half> %i105, <1 x half> %in) {
; CHECK-LABEL: @cmp_select
; CHECL: fcmge
newFuncRoot:
  %i179 = fcmp uno <1 x half> %i105, zeroinitializer
  %i180 = select <1 x i1> %i179, <1 x half> %in, <1 x half> %i105
  ret <1 x half> %i180
}
