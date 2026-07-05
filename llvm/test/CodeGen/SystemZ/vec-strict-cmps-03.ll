; Test signaling vector floating-point comparisons on z13.
; Note that these must be scalarized as we do not have native instructions.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test v4f32.
define <4 x i32> @f1(<4 x float> %val1, <4 x float> %val2) #0 {
; CHECK-LABEL: f1:
; CHECK: kebr
; CHECK: kebr
; CHECK: kebr
; CHECK: kebr
; CHECK: br %r14
  %cmp = call <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(
                                               <4 x float> %val1, <4 x float> %val2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <4 x i1> %cmp to <4 x i32>
  ret <4 x i32> %ret
}

; Test v2f64.
define <2 x i64> @f2(<2 x i64> %dummy, <2 x double> %val1, <2 x double> %val2) #0 {
; CHECK-LABEL: f2:
; CHECK: {{kdbr|wfkdb}}
; CHECK: {{kdbr|wfkdb}}
; CHECK: br %r14
  %cmp = call <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(
                                               <2 x double> %val1, <2 x double> %val2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %ret = sext <2 x i1> %cmp to <2 x i64>
  ret <2 x i64> %ret
}

; Test an f64 comparison that uses vector registers.
define i64 @f3(i64 %a, i64 %b, double %f1, <2 x double> %vec) #0 {
; CHECK-LABEL: f3:
; CHECK: wfkdb %f0, %v24
; CHECK-NEXT: locgrne %r2, %r3
; CHECK: br %r14
  %f2 = extractelement <2 x double> %vec, i32 0
  %cond = call i1 @llvm.experimental.constrained.fcmps.f64(
                                               double %f1, double %f2,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  %res = select i1 %cond, i64 %a, i64 %b
  ret i64 %res
}

attributes #0 = { strictfp }

declare <4 x i1> @llvm.experimental.constrained.fcmps.v4f32(<4 x float>, <4 x float>, metadata, metadata)
declare <2 x i1> @llvm.experimental.constrained.fcmps.v2f64(<2 x double>, <2 x double>, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f64(double, double, metadata, metadata)

