; RUN: opt < %s -S -p vector-combine | FileCheck %s

define <2 x i1> @test_icmp_eq(i32 %arg) {
; CHECK-LABEL: define <2 x i1> @test_icmp_eq(
; CHECK-SAME: i32 [[Arg:%.*]]) {
; CHECK-NEXT:  [[ScalarCmp:%.*]] = icmp eq i32 [[Arg]], 3
; CHECK-NEXT:  [[Insert:%.*]] = insertelement <2 x i1> splat (i1 true), i1 [[ScalarCmp]], i64 0
; CHECK-NEXT:  ret <2 x i1> [[Insert]]
  %vec0 = insertelement <2 x i32> zeroinitializer, i32 %arg, i64 0
  %vec1 = insertelement <2 x i32> zeroinitializer, i32 3, i64 0
  %vecCmp = icmp eq <2 x i32> %vec0, %vec1
  ret <2 x i1> %vecCmp
}

define <2 x i1> @test_icmp_slt(i32 %arg) {
; CHECK-LABEL: define <2 x i1> @test_icmp_slt(
; CHECK-SAME: i32 [[Arg:%.*]]) {
; CHECK-NEXT:  [[ScalarCmp:%.*]] = icmp slt i32 [[Arg]], 3
; CHECK-NEXT:  [[Insert:%.*]] = insertelement <2 x i1> zeroinitializer, i1 [[ScalarCmp]], i64 0
; CHECK-NEXT:  ret <2 x i1> [[Insert]]
  %vec0 = insertelement <2 x i32> zeroinitializer, i32 %arg, i64 0
  %vec1 = insertelement <2 x i32> zeroinitializer, i32 3, i64 0
  %vecCmp = icmp slt <2 x i32> %vec0, %vec1
  ret <2 x i1> %vecCmp
}

define <2 x i1> @test_fcmp_oeq(float %arg) {
; CHECK-LABEL: define <2 x i1> @test_fcmp_oeq(
; CHECK-SAME: float [[Arg:%.*]]) {
; CHECK-NEXT:  [[ScalarCmp:%.*]] = fcmp oeq float [[Arg]], 3.000000e+00
; CHECK-NEXT:  [[Insert:%.*]] = insertelement <2 x i1> splat (i1 true), i1 [[ScalarCmp]], i64 0
; CHECK-NEXT:  ret <2 x i1> [[Insert]]
  %vec0 = insertelement <2 x float> zeroinitializer, float %arg, i64 0
  %vec1 = insertelement <2 x float> zeroinitializer, float 3.000000e+00, i64 0
  %vecCmp = fcmp oeq <2 x float> %vec0, %vec1
  ret <2 x i1> %vecCmp
}

define <2 x i1> @test_fcmp_ult(float %arg) {
; CHECK-LABEL: define <2 x i1> @test_fcmp_ult(
; CHECK-SAME: float [[Arg:%.*]]) {
; CHECK-NEXT:  [[ScalarCmp:%.*]] = fcmp fast ult float [[Arg]], 3.000000e+00
; CHECK-NEXT:  [[Insert:%.*]] = insertelement <2 x i1> zeroinitializer, i1 [[ScalarCmp]], i64 0
; CHECK-NEXT:  ret <2 x i1> [[Insert]]
  %vec0 = insertelement <2 x float> zeroinitializer, float %arg, i64 0
  %vec1 = insertelement <2 x float> zeroinitializer, float 3.000000e+00, i64 0
  %vecCmp = fcmp fast ult <2 x float> %vec0, %vec1
  ret <2 x i1> %vecCmp
}

