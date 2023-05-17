; Test strict vector maximum on z14.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z14 | FileCheck %s

declare double @llvm.experimental.constrained.maxnum.f64(double, double, metadata)
declare <2 x double> @llvm.experimental.constrained.maxnum.v2f64(<2 x double>, <2 x double>, metadata)
declare double @llvm.experimental.constrained.maximum.f64(double, double, metadata)
declare <2 x double> @llvm.experimental.constrained.maximum.v2f64(<2 x double>, <2 x double>, metadata)

declare float @llvm.experimental.constrained.maxnum.f32(float, float, metadata)
declare <4 x float> @llvm.experimental.constrained.maxnum.v4f32(<4 x float>, <4 x float>, metadata)
declare float @llvm.experimental.constrained.maximum.f32(float, float, metadata)
declare <4 x float> @llvm.experimental.constrained.maximum.v4f32(<4 x float>, <4 x float>, metadata)

declare fp128 @llvm.experimental.constrained.maxnum.f128(fp128, fp128, metadata)
declare fp128 @llvm.experimental.constrained.maximum.f128(fp128, fp128, metadata)

; Test the f64 maxnum intrinsic.
define double @f1(double %dummy, double %val1, double %val2) #0 {
; CHECK-LABEL: f1:
; CHECK: wfmaxdb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call double @llvm.experimental.constrained.maxnum.f64(
                        double %val1, double %val2,
                        metadata !"fpexcept.strict") #0
  ret double %ret
}

; Test the v2f64 maxnum intrinsic.
define <2 x double> @f2(<2 x double> %dummy, <2 x double> %val1,
                        <2 x double> %val2) #0 {
; CHECK-LABEL: f2:
; CHECK: vfmaxdb %v24, %v26, %v28, 4
; CHECK: br %r14
  %ret = call <2 x double> @llvm.experimental.constrained.maxnum.v2f64(
                        <2 x double> %val1, <2 x double> %val2,
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %ret
}

; Test the f32 maxnum intrinsic.
define float @f3(float %dummy, float %val1, float %val2) #0 {
; CHECK-LABEL: f3:
; CHECK: wfmaxsb %f0, %f2, %f4, 4
; CHECK: br %r14
  %ret = call float @llvm.experimental.constrained.maxnum.f32(
                        float %val1, float %val2,
                        metadata !"fpexcept.strict") #0
  ret float %ret
}

; Test the v4f32 maxnum intrinsic.
define <4 x float> @f4(<4 x float> %dummy, <4 x float> %val1,
                       <4 x float> %val2) #0 {
; CHECK-LABEL: f4:
; CHECK: vfmaxsb %v24, %v26, %v28, 4
; CHECK: br %r14
  %ret = call <4 x float> @llvm.experimental.constrained.maxnum.v4f32(
                        <4 x float> %val1, <4 x float> %val2,
                        metadata !"fpexcept.strict") #0
  ret <4 x float> %ret
}

; Test the f128 maxnum intrinsic.
define void @f5(ptr %ptr1, ptr %ptr2, ptr %dst) #0 {
; CHECK-LABEL: f5:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfmaxxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], 4
; CHECK: vst [[RES]], 0(%r4)
; CHECK: br %r14
  %val1 = load fp128, ptr %ptr1
  %val2 = load fp128, ptr %ptr2
  %res = call fp128 @llvm.experimental.constrained.maxnum.f128(
                        fp128 %val1, fp128 %val2,
                        metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

; Test the f64 maximum intrinsic.
define double @f11(double %dummy, double %val1, double %val2) #0 {
; CHECK-LABEL: f11:
; CHECK: wfmaxdb %f0, %f2, %f4, 1
; CHECK: br %r14
  %ret = call double @llvm.experimental.constrained.maximum.f64(
                        double %val1, double %val2,
                        metadata !"fpexcept.strict") #0
  ret double %ret
}

; Test the v2f64 maximum intrinsic.
define <2 x double> @f12(<2 x double> %dummy, <2 x double> %val1,
                         <2 x double> %val2) #0 {
; CHECK-LABEL: f12:
; CHECK: vfmaxdb %v24, %v26, %v28, 1
; CHECK: br %r14
  %ret = call <2 x double> @llvm.experimental.constrained.maximum.v2f64(
                        <2 x double> %val1, <2 x double> %val2,
                        metadata !"fpexcept.strict") #0
  ret <2 x double> %ret
}

; Test the f32 maximum intrinsic.
define float @f13(float %dummy, float %val1, float %val2) #0 {
; CHECK-LABEL: f13:
; CHECK: wfmaxsb %f0, %f2, %f4, 1
; CHECK: br %r14
  %ret = call float @llvm.experimental.constrained.maximum.f32(
                        float %val1, float %val2,
                        metadata !"fpexcept.strict") #0
  ret float %ret
}

; Test the v4f32 maximum intrinsic.
define <4 x float> @f14(<4 x float> %dummy, <4 x float> %val1,
                        <4 x float> %val2) #0 {
; CHECK-LABEL: f14:
; CHECK: vfmaxsb %v24, %v26, %v28, 1
; CHECK: br %r14
  %ret = call <4 x float> @llvm.experimental.constrained.maximum.v4f32(
                        <4 x float> %val1, <4 x float> %val2,
                        metadata !"fpexcept.strict") #0
  ret <4 x float> %ret
}

; Test the f128 maximum intrinsic.
define void @f15(ptr %ptr1, ptr %ptr2, ptr %dst) #0 {
; CHECK-LABEL: f15:
; CHECK-DAG: vl [[REG1:%v[0-9]+]], 0(%r2)
; CHECK-DAG: vl [[REG2:%v[0-9]+]], 0(%r3)
; CHECK: wfmaxxb [[RES:%v[0-9]+]], [[REG1]], [[REG2]], 1
; CHECK: vst [[RES]], 0(%r4)
; CHECK: br %r14
  %val1 = load fp128, ptr %ptr1
  %val2 = load fp128, ptr %ptr2
  %res = call fp128 @llvm.experimental.constrained.maximum.f128(
                        fp128 %val1, fp128 %val2,
                        metadata !"fpexcept.strict") #0
  store fp128 %res, ptr %dst
  ret void
}

attributes #0 = { strictfp }
