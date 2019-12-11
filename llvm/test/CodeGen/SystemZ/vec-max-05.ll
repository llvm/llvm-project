; Test vector maximum on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare double @fmax(double, double)
declare double @llvm.maxnum.f64(double, double)
declare <2 x double> @llvm.maxnum.v2f64(<2 x double>, <2 x double>)
declare double @llvm.maximum.f64(double, double)
declare <2 x double> @llvm.maximum.v2f64(<2 x double>, <2 x double>)

declare float @fmaxf(float, float)
declare float @llvm.maxnum.f32(float, float)
declare <4 x float> @llvm.maxnum.v4f32(<4 x float>, <4 x float>)
declare float @llvm.maximum.f32(float, float)
declare <4 x float> @llvm.maximum.v4f32(<4 x float>, <4 x float>)

declare fp128 @fmaxl(fp128, fp128)
declare fp128 @llvm.maxnum.f128(fp128, fp128)
declare fp128 @llvm.maximum.f128(fp128, fp128)

; Test the fmax library function.
define double @f1(double %dummy, double %val1, double %val2) {
; CHECK-LABEL: f1:
; CHECK: wfmaxdb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call double @fmax(double %val1, double %val2) readnone
  ret double %ret
}

; Test the f64 maxnum intrinsic.
define double @f2(double %dummy, double %val1, double %val2) {
; CHECK-LABEL: f2:
; CHECK: wfmaxdb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call double @llvm.maxnum.f64(double %val1, double %val2)
  ret double %ret
}

; Test the f64 maximum intrinsic.
define double @f3(double %dummy, double %val1, double %val2) {
; CHECK-LABEL: f3:
; CHECK: wfmaxdb %f0, %f2, %f4, 1
; CHECK: br %r14
  %ret = call double @llvm.maximum.f64(double %val1, double %val2)
  ret double %ret
}

; Test a f64 constant compare/select resulting in maxnum.
define double @f4(double %dummy, double %val) {
; CHECK-LABEL: f4:
; CHECK: lzdr [[REG:%f[0-9]+]]
; CHECK: wfmaxdb %f0, %f2, [[REG]], 4
; CHECK: br %r14
  %cmp = fcmp ogt double %val, 0.0
  %ret = select i1 %cmp, double %val, double 0.0
  ret double %ret
}

; Test a f64 constant compare/select resulting in maximum.
define double @f5(double %dummy, double %val) {
; CHECK-LABEL: f5:
; CHECK: lzdr [[REG:%f[0-9]+]]
; CHECK: wfmaxdb %f0, %f2, [[REG]], 1
; CHECK: br %r14
  %cmp = fcmp ugt double %val, 0.0
  %ret = select i1 %cmp, double %val, double 0.0
  ret double %ret
}

; Test the v2f64 maxnum intrinsic.
define <2 x double> @f6(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f6:
; CHECK: vfmaxdb %v24, %v26, %v28, 4
; CHECK: br %r14
  %ret = call <2 x double> @llvm.maxnum.v2f64(<2 x double> %val1, <2 x double> %val2)
  ret <2 x double> %ret
}

; Test the v2f64 maximum intrinsic.
define <2 x double> @f7(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) {
; CHECK-LABEL: f7:
; CHECK: vfmaxdb %v24, %v26, %v28, 1
; CHECK: br %r14
  %ret = call <2 x double> @llvm.maximum.v2f64(<2 x double> %val1, <2 x double> %val2)
  ret <2 x double> %ret
}

; Test the fmaxf library function.
define float @f11(float %dummy, float %val1, float %val2) {
; CHECK-LABEL: f11:
; CHECK: wfmaxsb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call float @fmaxf(float %val1, float %val2) readnone
  ret float %ret
}

; Test the f32 maxnum intrinsic.
define float @f12(float %dummy, float %val1, float %val2) {
; CHECK-LABEL: f12:
; CHECK: wfmaxsb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call float @llvm.maxnum.f32(float %val1, float %val2)
  ret float %ret
}

; Test the f32 maximum intrinsic.
define float @f13(float %dummy, float %val1, float %val2) {
; CHECK-LABEL: f13:
; CHECK: wfmaxsb %f0, %f2, %f4, 1
; CHECK: br %r14
  %ret = call float @llvm.maximum.f32(float %val1, float %val2)
  ret float %ret
}

; Test a f32 constant compare/select resulting in maxnum.
define float @f14(float %dummy, float %val) {
; CHECK-LABEL: f14:
; CHECK: lzer [[REG:%f[0-9]+]]
; CHECK: wfmaxsb %f0, %f2, [[REG]], 4
; CHECK: br %r14
  %cmp = fcmp ogt float %val, 0.0
  %ret = select i1 %cmp, float %val, float 0.0
  ret float %ret
}

; Test a f32 constant compare/select resulting in maximum.
define float @f15(float %dummy, float %val) {
; CHECK-LABEL: f15:
; CHECK: lzer [[REG:%f[0-9]+]]
; CHECK: wfmaxsb %f0, %f2, [[REG]], 1
; CHECK: br %r14
  %cmp = fcmp ugt float %val, 0.0
  %ret = select i1 %cmp, float %val, float 0.0
  ret float %ret
}

; Test the v4f32 maxnum intrinsic.
define <4 x float> @f16(<4 x float> %dummy, <4 x float> %val1,
                        <4 x float> %val2) {
; CHECK-LABEL: f16:
; CHECK: vfmaxsb %v24, %v26, %v28, 4
; CHECK: br %r14
  %ret = call <4 x float> @llvm.maxnum.v4f32(<4 x float> %val1, <4 x float> %val2)
  ret <4 x float> %ret
}

; Test the v4f32 maximum intrinsic.
define <4 x float> @f17(<4 x float> %dummy, <4 x float> %val1,
                        <4 x float> %val2) {
; CHECK-LABEL: f17:
; CHECK: vfmaxsb %v24, %v26, %v28, 1
; CHECK: br %r14
  %ret = call <4 x float> @llvm.maximum.v4f32(<4 x float> %val1, <4 x float> %val2)
  ret <4 x float> %ret
}

; Test the fmaxl library function.
define void @f21(fp128 *%ptr1, fp128 *%ptr2, fp128 *%dst) {
; CHECK-LABEL: f21:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfmaxxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], 4
; CHECK: vst [[RES]], 0(%r4)
; CHECK: br %r14
  %val1 = load fp128, fp128* %ptr1
  %val2 = load fp128, fp128* %ptr2
  %res = call fp128 @fmaxl(fp128 %val1, fp128 %val2) readnone
  store fp128 %res, fp128* %dst
  ret void
}

; Test the f128 maxnum intrinsic.
define void @f22(fp128 *%ptr1, fp128 *%ptr2, fp128 *%dst) {
; CHECK-LABEL: f22:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfmaxxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], 4
; CHECK: vst [[RES]], 0(%r4)
; CHECK: br %r14
  %val1 = load fp128, fp128* %ptr1
  %val2 = load fp128, fp128* %ptr2
  %res = call fp128 @llvm.maxnum.f128(fp128 %val1, fp128 %val2)
  store fp128 %res, fp128* %dst
  ret void
}

; Test the f128 maximum intrinsic.
define void @f23(fp128 *%ptr1, fp128 *%ptr2, fp128 *%dst) {
; CHECK-LABEL: f23:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfmaxxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], 1
; CHECK: vst [[RES]], 0(%r4)
; CHECK: br %r14
  %val1 = load fp128, fp128* %ptr1
  %val2 = load fp128, fp128* %ptr2
  %res = call fp128 @llvm.maximum.f128(fp128 %val1, fp128 %val2)
  store fp128 %res, fp128* %dst
  ret void
}

; Test a f128 constant compare/select resulting in maxnum.
define void @f24(fp128 *%ptr, fp128 *%dst) {
; CHECK-LABEL: f24:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vzero [[REG2:%v[0-9]+]]
; CHECK: wfmaxxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], 4
; CHECK: vst [[RES]], 0(%r3)
; CHECK: br %r14
  %val = load fp128, fp128* %ptr
  %cmp = fcmp ogt fp128 %val, 0xL00000000000000000000000000000000
  %res = select i1 %cmp, fp128 %val, fp128 0xL00000000000000000000000000000000
  store fp128 %res, fp128* %dst
  ret void
}

; Test a f128 constant compare/select resulting in maximum.
define void @f25(fp128 *%ptr, fp128 *%dst) {
; CHECK-LABEL: f25:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vzero [[REG2:%v[0-9]+]]
; CHECK: wfmaxxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], 1
; CHECK: vst [[RES]], 0(%r3)
; CHECK: br %r14
  %val = load fp128, fp128* %ptr
  %cmp = fcmp ugt fp128 %val, 0xL00000000000000000000000000000000
  %res = select i1 %cmp, fp128 %val, fp128 0xL00000000000000000000000000000000
  store fp128 %res, fp128* %dst
  ret void
}

