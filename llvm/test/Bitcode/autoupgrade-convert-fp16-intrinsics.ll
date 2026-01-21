; RUN: llvm-as < %s | llvm-dis | FileCheck %s


define i16 @convert_to_fp16__f32(float %src) {
; CHECK-LABEL: define i16 @convert_to_fp16__f32(
; CHECK-SAME: float [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = fptrunc float [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast half [[TMP1]] to i16
; CHECK-NEXT:    ret i16 [[TMP2]]
;
  %result = call i16 @llvm.convert.to.fp16.f32(float %src)
  ret i16 %result
}

define i16 @convert_to_fp16__f64(double %src) {
; CHECK-LABEL: define i16 @convert_to_fp16__f64(
; CHECK-SAME: double [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = fptrunc double [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast half [[TMP1]] to i16
; CHECK-NEXT:    ret i16 [[TMP2]]
;
  %result = call i16 @llvm.convert.to.fp16.f64(double %src)
  ret i16 %result
}

define i16 @convert_to_fp16__fp128(fp128 %src) {
; CHECK-LABEL: define i16 @convert_to_fp16__fp128(
; CHECK-SAME: fp128 [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = fptrunc fp128 [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast half [[TMP1]] to i16
; CHECK-NEXT:    ret i16 [[TMP2]]
;
  %result = call i16 @llvm.convert.to.fp16.f128(fp128 %src)
  ret i16 %result
}

define i16 @convert_to_fp16__x86_fp80(x86_fp80 %src) {
; CHECK-LABEL: define i16 @convert_to_fp16__x86_fp80(
; CHECK-SAME: x86_fp80 [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = fptrunc x86_fp80 [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast half [[TMP1]] to i16
; CHECK-NEXT:    ret i16 [[TMP2]]
;
  %result = call i16 @llvm.convert.to.fp16.f80(x86_fp80 %src)
  ret i16 %result
}

define i16 @convert_to_fp16__ppc_fp128(ppc_fp128 %src) {
; CHECK-LABEL: define i16 @convert_to_fp16__ppc_fp128(
; CHECK-SAME: ppc_fp128 [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = fptrunc ppc_fp128 [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = bitcast half [[TMP1]] to i16
; CHECK-NEXT:    ret i16 [[TMP2]]
;
  %result = call i16 @llvm.convert.to.fp16.ppcf128(ppc_fp128 %src)
  ret i16 %result
}

define float @convert_from_fp16__f32(i16 %src) {
; CHECK-LABEL: define float @convert_from_fp16__f32(
; CHECK-SAME: i16 [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i16 [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = fpext half [[TMP1]] to float
; CHECK-NEXT:    ret float [[TMP2]]
;
  %result = call float @llvm.convert.from.fp16.f32(i16 %src)
  ret float %result
}

define double @convert_from_fp16__f64(i16 %src) {
; CHECK-LABEL: define double @convert_from_fp16__f64(
; CHECK-SAME: i16 [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i16 [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = fpext half [[TMP1]] to double
; CHECK-NEXT:    ret double [[TMP2]]
;
  %result = call double @llvm.convert.from.fp16.f64(i16 %src)
  ret double %result
}

define fp128 @convert_from_fp16__fp128(i16 %src) {
; CHECK-LABEL: define fp128 @convert_from_fp16__fp128(
; CHECK-SAME: i16 [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i16 [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = fpext half [[TMP1]] to fp128
; CHECK-NEXT:    ret fp128 [[TMP2]]
;
  %result = call fp128 @llvm.convert.from.fp16.f128(i16 %src)
  ret fp128 %result
}

define x86_fp80 @convert_from_fp16__x86_fp80(i16 %src) {
; CHECK-LABEL: define x86_fp80 @convert_from_fp16__x86_fp80(
; CHECK-SAME: i16 [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i16 [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = fpext half [[TMP1]] to x86_fp80
; CHECK-NEXT:    ret x86_fp80 [[TMP2]]
;
  %result = call x86_fp80 @llvm.convert.from.fp16.f80(i16 %src)
  ret x86_fp80 %result
}

define ppc_fp128 @convert_from_fp16__ppc_fp128_fp80(i16 %src) {
; CHECK-LABEL: define ppc_fp128 @convert_from_fp16__ppc_fp128_fp80(
; CHECK-SAME: i16 [[SRC:%.*]]) {
; CHECK-NEXT:    [[TMP1:%.*]] = bitcast i16 [[SRC]] to half
; CHECK-NEXT:    [[TMP2:%.*]] = fpext half [[TMP1]] to ppc_fp128
; CHECK-NEXT:    ret ppc_fp128 [[TMP2]]
;
  %result = call ppc_fp128 @llvm.convert.from.fp16.ppcf128(i16 %src)
  ret ppc_fp128 %result
}


declare i16 @llvm.convert.to.fp16.f32(float) #0
declare i16 @llvm.convert.to.fp16.f64(double) #0
declare i16 @llvm.convert.to.fp16.f128(fp128) #0
declare i16 @llvm.convert.to.fp16.f80(x86_fp80) #0
declare i16 @llvm.convert.to.fp16.ppcf128(ppc_fp128) #0

declare float @llvm.convert.from.fp16.f32(i16) #0
declare double @llvm.convert.from.fp16.f64(i16) #0
declare fp128 @llvm.convert.from.fp16.f128(i16) #0
declare x86_fp80 @llvm.convert.from.fp16.f80(i16) #0
declare ppc_fp128 @llvm.convert.from.fp16.ppcf128(i16) #0

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(none) }
