; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.9-library %s | FileCheck %s --check-prefixes=CHECK,SM69CHECK
; RUN: opt -S -dxil-intrinsic-expansion -scalarizer -dxil-op-lower -mtriple=dxil-pc-shadermodel6.8-library %s | FileCheck %s --check-prefixes=CHECK,SMOLDCHECK


define noundef i1 @isnegzero(float noundef %a) {
; CHECK-LABEL: define noundef i1 @isnegzero(
; CHECK-SAME: float noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = bitcast float [[A]] to i32
; CHECK-NEXT:    [[IS_FPCLASS_NEGZERO:%.*]] = icmp eq i32 [[TMP0]], -2147483648
; CHECK-NEXT:    ret i1 [[IS_FPCLASS_NEGZERO]]
;
entry:
  %0 = call i1 @llvm.is.fpclass.f32(float %a, i32 32)
  ret i1 %0
}

define noundef <2 x i1> @isnegzerov2(<2 x float> noundef %a) {
; CHECK-LABEL: define noundef <2 x i1> @isnegzerov2(
; CHECK-SAME: <2 x float> noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[A_I0:%.*]] = extractelement <2 x float> [[A]], i64 0
; CHECK-NEXT:    [[DOTI0:%.*]] = bitcast float [[A_I0]] to i32
; CHECK-NEXT:    [[A_I1:%.*]] = extractelement <2 x float> [[A]], i64 1
; CHECK-NEXT:    [[DOTI1:%.*]] = bitcast float [[A_I1]] to i32
; CHECK-NEXT:    [[IS_FPCLASS_NEGZERO_I0:%.*]] = icmp eq i32 [[DOTI0]], -2147483648
; CHECK-NEXT:    [[IS_FPCLASS_NEGZERO_I1:%.*]] = icmp eq i32 [[DOTI1]], -2147483648
; CHECK-NEXT:    [[IS_FPCLASS_NEGZERO_UPTO0:%.*]] = insertelement <2 x i1> poison, i1 [[IS_FPCLASS_NEGZERO_I0]], i64 0
; CHECK-NEXT:    [[IS_FPCLASS_NEGZERO:%.*]] = insertelement <2 x i1> [[IS_FPCLASS_NEGZERO_UPTO0]], i1 [[IS_FPCLASS_NEGZERO_I1]], i64 1
; CHECK-NEXT:    ret <2 x i1> [[IS_FPCLASS_NEGZERO]]
;
entry:
  %0 = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %a, i32 32)
  ret <2 x i1> %0
}

define noundef i1 @isnan(float noundef %a) {
; CHECK-LABEL: define noundef i1 @isnan(
; CHECK-SAME: float noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 8, float [[A]]) #[[ATTR0:[0-9]+]]
; CHECK-NEXT:    ret i1 [[TMP0]]
;
entry:
  %0 = call i1 @llvm.is.fpclass.f32(float %a, i32 3)
  ret i1 %0
}

define noundef <2 x i1> @isnanv2(<2 x float> noundef %a) {
; CHECK-LABEL: define noundef <2 x i1> @isnanv2(
; CHECK-SAME: <2 x float> noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[A_I0:%.*]] = extractelement <2 x float> [[A]], i64 0
; CHECK-NEXT:    [[DOTI02:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 8, float [[A_I0]]) #[[ATTR0]]
; CHECK-NEXT:    [[A_I1:%.*]] = extractelement <2 x float> [[A]], i64 1
; CHECK-NEXT:    [[DOTI11:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 8, float [[A_I1]]) #[[ATTR0]]
; CHECK-NEXT:    [[DOTUPTO0:%.*]] = insertelement <2 x i1> poison, i1 [[DOTI02]], i64 0
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <2 x i1> [[DOTUPTO0]], i1 [[DOTI11]], i64 1
; CHECK-NEXT:    ret <2 x i1> [[TMP0]]
;
entry:
  %0 = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %a, i32 3)
  ret <2 x i1> %0
}

define noundef i1 @isinf(float noundef %a) {
; CHECK-LABEL: define noundef i1 @isinf(
; CHECK-SAME: float noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 9, float [[A]]) #[[ATTR0]]
; CHECK-NEXT:    ret i1 [[TMP0]]
;
entry:
  %0 = call i1 @llvm.is.fpclass.f32(float %a, i32 516)
  ret i1 %0
}

define noundef i1 @isinfh(half noundef %a) {
; CHECK-LABEL: define noundef i1 @isinfh(
; CHECK-SAME: half noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; SM69CHECK-NEXT:    [[ISINF:%.*]] = call i1 @dx.op.isSpecialFloat.f16(i32 9, half [[A]]) #[[ATTR0]]
; SMOLDCHECK-NEXT: [[BITCAST:%.*]] = bitcast half %a to i16
; SMOLDCHECK-NEXT: [[CMPHIGH:%.*]] = icmp eq i16 [[BITCAST]], 31744
; SMOLDCHECK-NEXT: [[CMPLOW:%.*]] = icmp eq i16 [[BITCAST]], -1024
; SMOLDCHECK-NEXT: [[OR:%.*]] = or i1 [[CMPHIGH]], [[CMPLOW]]
; SMOLDCHECK-NEXT:   ret i1 [[OR]]
; SM69CHECK-NEXT:    ret i1 [[ISINF]]
;
entry:
  %0 = call i1 @llvm.is.fpclass.f16(half %a, i32 516)
  ret i1 %0
}

define noundef <2 x i1> @isinfv2(<2 x float> noundef %a) {
; CHECK-LABEL: define noundef <2 x i1> @isinfv2(
; CHECK-SAME: <2 x float> noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[A_I0:%.*]] = extractelement <2 x float> [[A]], i64 0
; CHECK-NEXT:    [[DOTI02:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 9, float [[A_I0]]) #[[ATTR0]]
; CHECK-NEXT:    [[A_I1:%.*]] = extractelement <2 x float> [[A]], i64 1
; CHECK-NEXT:    [[DOTI11:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 9, float [[A_I1]]) #[[ATTR0]]
; CHECK-NEXT:    [[DOTUPTO0:%.*]] = insertelement <2 x i1> poison, i1 [[DOTI02]], i64 0
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <2 x i1> [[DOTUPTO0]], i1 [[DOTI11]], i64 1
; CHECK-NEXT:    ret <2 x i1> [[TMP0]]
;
entry:
  %0 = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %a, i32 516)
  ret <2 x i1> %0
}

define noundef i1 @isfinite(float noundef %a) {
; CHECK-LABEL: define noundef i1 @isfinite(
; CHECK-SAME: float noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 10, float [[A]]) #[[ATTR0]]
; CHECK-NEXT:    ret i1 [[TMP0]]
;
entry:
  %0 = call i1 @llvm.is.fpclass.f32(float %a, i32 504)
  ret i1 %0
}

define noundef <2 x i1> @isfinitev2(<2 x float> noundef %a) {
; CHECK-LABEL: define noundef <2 x i1> @isfinitev2(
; CHECK-SAME: <2 x float> noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[A_I0:%.*]] = extractelement <2 x float> [[A]], i64 0
; CHECK-NEXT:    [[DOTI02:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 10, float [[A_I0]]) #[[ATTR0]]
; CHECK-NEXT:    [[A_I1:%.*]] = extractelement <2 x float> [[A]], i64 1
; CHECK-NEXT:    [[DOTI11:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 10, float [[A_I1]]) #[[ATTR0]]
; CHECK-NEXT:    [[DOTUPTO0:%.*]] = insertelement <2 x i1> poison, i1 [[DOTI02]], i64 0
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <2 x i1> [[DOTUPTO0]], i1 [[DOTI11]], i64 1
; CHECK-NEXT:    ret <2 x i1> [[TMP0]]
;
entry:
  %0 = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %a, i32 504)
  ret <2 x i1> %0
}

define noundef i1 @isnormal(float noundef %a) {
; CHECK-LABEL: define noundef i1 @isnormal(
; CHECK-SAME: float noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[TMP0:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 11, float [[A]]) #[[ATTR0]]
; CHECK-NEXT:    ret i1 [[TMP0]]
;
entry:
  %0 = call i1 @llvm.is.fpclass.f32(float %a, i32 264)
  ret i1 %0
}

define noundef <2 x i1> @isnormalv2(<2 x float> noundef %a) {
; CHECK-LABEL: define noundef <2 x i1> @isnormalv2(
; CHECK-SAME: <2 x float> noundef [[A:%.*]]) {
; CHECK-NEXT:  [[ENTRY:.*:]]
; CHECK-NEXT:    [[A_I0:%.*]] = extractelement <2 x float> [[A]], i64 0
; CHECK-NEXT:    [[DOTI02:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 11, float [[A_I0]]) #[[ATTR0]]
; CHECK-NEXT:    [[A_I1:%.*]] = extractelement <2 x float> [[A]], i64 1
; CHECK-NEXT:    [[DOTI11:%.*]] = call i1 @dx.op.isSpecialFloat.f32(i32 11, float [[A_I1]]) #[[ATTR0]]
; CHECK-NEXT:    [[DOTUPTO0:%.*]] = insertelement <2 x i1> poison, i1 [[DOTI02]], i64 0
; CHECK-NEXT:    [[TMP0:%.*]] = insertelement <2 x i1> [[DOTUPTO0]], i1 [[DOTI11]], i64 1
; CHECK-NEXT:    ret <2 x i1> [[TMP0]]
;
entry:
  %0 = call <2 x i1> @llvm.is.fpclass.v2f32(<2 x float> %a, i32 264)
  ret <2 x i1> %0
}

declare i1 @llvm.is.fpclass.f32(float, i32 immarg)
declare <2 x i1> @llvm.is.fpclass.v2f32(<2 x float>, i32 immarg)
