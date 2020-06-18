; Verify that floating-point strict signaling compares cannot be omitted
; even if CC already has the right value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   -enable-misched=0 -no-integrated-as | FileCheck %s
;
; We need -enable-misched=0 to make sure f12 and following routines really
; test the compare elimination pass.


declare float @llvm.fabs.f32(float %f)

; Test addition followed by EQ, which could use the CC result of the addition.
define float @f1(float %a, float %b, float *%dest) #0 {
; CHECK-LABEL: f1:
; CHECK-DAG: aebr %f0, %f2
; CHECK-DAG: lzer [[REG:%f[0-9]+]]
; CHECK-NEXT: kebr %f0, [[REG]]
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(
                                               float %res, float 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, float *%dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD POSITIVE.
define float @f6(float %dummy, float %a, float *%dest) #0 {
; CHECK-LABEL: f6:
; CHECK-DAG: lpdfr %f0, %f2
; CHECK-DAG: lzer [[REG:%f[0-9]+]]
; CHECK-NEXT: kebr %f0, [[REG]]
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.fabs.f32(float %a) #0
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(
                                               float %res, float 0.0,
                                               metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, float *%dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD NEGATIVE.
define float @f7(float %dummy, float %a, float *%dest) #0 {
; CHECK-LABEL: f7:
; CHECK-DAG: lndfr %f0, %f2
; CHECK-DAG: lzer [[REG:%f[0-9]+]]
; CHECK-NEXT: kebr %f0, [[REG]]
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %abs = call float @llvm.fabs.f32(float %a) #0
  %res = fneg float %abs
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(
                                               float %res, float 0.0,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, float *%dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD COMPLEMENT.
define float @f8(float %dummy, float %a, float *%dest) #0 {
; CHECK-LABEL: f8:
; CHECK-DAG: lcdfr %f0, %f2
; CHECK-DAG: lzer [[REG:%f[0-9]+]]
; CHECK-NEXT: kebr %f0, [[REG]]
; CHECK-NEXT: bler %r14
; CHECK: br %r14
entry:
  %res = fneg float %a
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(
                                               float %res, float 0.0,
                                               metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, float *%dest
  br label %exit

exit:
  ret float %res
}

; Test that LER does not get converted to LTEBR.
define float @f12(float %dummy, float %val) #0 {
; CHECK-LABEL: f12:
; CHECK: ler %f0, %f2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f0
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: lzer [[REG:%f[0-9]+]]
; CHECK-NEXT: kebr %f2, [[REG]]
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %ret = call float asm "blah $1", "=f,{f0}"(float %val) #0
  %cmp = call i1 @llvm.experimental.constrained.fcmps.f32(
                                               float %val, float 0.0,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  call void asm sideeffect "blah", ""() #0
  br label %exit

exit:
  ret float %ret
}

attributes #0 = { strictfp }

declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fsub.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fmul.f32(float, float, metadata, metadata)
declare float @llvm.experimental.constrained.fdiv.f32(float, float, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f32(float, float, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f64(double, double, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmps.f128(fp128, fp128, metadata, metadata)
