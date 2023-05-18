; Test that floating-point strict compares are omitted if CC already has the
; right value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 \
; RUN:   -enable-misched=0 -no-integrated-as | FileCheck %s
;
; We need -enable-misched=0 to make sure f12 and following routines really
; test the compare elimination pass.


declare float @llvm.fabs.f32(float %f)

; Test addition followed by EQ, which can use the CC result of the addition.
define float @f1(float %a, float %b, ptr %dest) #0 {
; CHECK-LABEL: f1:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; ...and again with LT.
define float @f2(float %a, float %b, ptr %dest) #0 {
; CHECK-LABEL: f2:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; ...and again with GT.
define float @f3(float %a, float %b, ptr %dest) #0 {
; CHECK-LABEL: f3:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; ...and again with UEQ.
define float @f4(float %a, float %b, ptr %dest) #0 {
; CHECK-LABEL: f4:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: bnlhr %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"ueq",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Subtraction also provides a zero-based CC value.
define float @f5(float %a, float %b, ptr %dest) #0 {
; CHECK-LABEL: f5:
; CHECK: seb %f0, 0(%r2)
; CHECK-NEXT: bnher %r14
; CHECK: br %r14
entry:
  %cur = load float, ptr %dest
  %res = call float @llvm.experimental.constrained.fsub.f32(
                        float %a, float %cur,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"ult",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD POSITIVE.  We cannot omit the LTEBR.
define float @f6(float %dummy, float %a, ptr %dest) #0 {
; CHECK-LABEL: f6:
; CHECK: lpdfr %f0, %f2
; CHECK-NEXT: ltebr %f1, %f0
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.fabs.f32(float %a) #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"ogt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD NEGATIVE.  We cannot omit the LTEBR.
define float @f7(float %dummy, float %a, ptr %dest) #0 {
; CHECK-LABEL: f7:
; CHECK: lndfr %f0, %f2
; CHECK-NEXT: ltebr %f1, %f0
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %abs = call float @llvm.fabs.f32(float %a) #0
  %res = fneg float %abs
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD COMPLEMENT.  We cannot omit the LTEBR.
define float @f8(float %dummy, float %a, ptr %dest) #0 {
; CHECK-LABEL: f8:
; CHECK: lcdfr %f0, %f2
; CHECK-NEXT: ltebr %f1, %f0
; CHECK-NEXT: bler %r14
; CHECK: br %r14
entry:
  %res = fneg float %a
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"ole",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Multiplication (for example) does not modify CC.
define float @f9(float %a, float %b, ptr %dest) #0 {
; CHECK-LABEL: f9:
; CHECK: meebr %f0, %f2
; CHECK-NEXT: ltebr %f1, %f0
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.experimental.constrained.fmul.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"one",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test a combination involving a CC-setting instruction followed by
; a non-CC-setting instruction.
define float @f10(float %a, float %b, float %c, ptr %dest) #0 {
; CHECK-LABEL: f10:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: debr %f0, %f4
; CHECK-NEXT: ltebr %f1, %f0
; CHECK-NEXT: bner %r14
; CHECK: br %r14
entry:
  %add = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %res = call float @llvm.experimental.constrained.fdiv.f32(
                        float %add, float %c,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"une",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test a case where CC is set based on a different register from the
; compare input.
define float @f11(float %a, float %b, float %c, ptr %dest1, ptr %dest2) #0 {
; CHECK-LABEL: f11:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: sebr %f4, %f0
; CHECK-DAG: ste %f4, 0(%r2)
; CHECK-DAG: ltebr %f1, %f0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %add = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %sub = call float @llvm.experimental.constrained.fsub.f32(
                        float %c, float %add,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  store float %sub, ptr %dest1
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %add, float 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %sub, ptr %dest2
  br label %exit

exit:
  ret float %add
}

; Test that LER does not get converted to LTEBR as %f0 is live after it.
define float @f12(float %dummy, float %val) #0 {
; CHECK-LABEL: f12:
; CHECK: ler %f0, %f2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f0
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ltebr %f1, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %ret = call float asm "blah $1", "=f,{f0}"(float %val) #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
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

; Test that LDR does not get converted to LTDBR as %f0 is live after it.
define double @f13(double %dummy, double %val) #0 {
; CHECK-LABEL: f13:
; CHECK: ldr %f0, %f2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f0
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ltdbr %f1, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %ret = call double asm "blah $1", "=f,{f0}"(double %val) #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %val, double 0.0,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  call void asm sideeffect "blah", ""() #0
  br label %exit

exit:
  ret double %ret
}

; Test that LXR does not get converted to LTXBR as %f4 is live after it.
define void @f14(ptr %ptr1, ptr %ptr2) #0 {
; CHECK-LABEL: f14:
; CHECK: lxr
; CHECK-NEXT: dxbr
; CHECK-NEXT: std
; CHECK-NEXT: std
; CHECK-NEXT: mxbr
; CHECK-NEXT: std
; CHECK-NEXT: std
; CHECK-NEXT: ltxbr
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %val1 = load fp128, ptr %ptr1
  %val2 = load fp128, ptr %ptr2
  %div = fdiv fp128 %val1, %val2
  store fp128 %div, ptr %ptr1
  %mul = fmul fp128 %val1, %val2
  store fp128 %mul, ptr %ptr2
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f128(
                                               fp128 %val1, fp128 0xL00000000000000000000000000000000,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  call void asm sideeffect "blah", ""() #0
  br label %exit

exit:
  ret void
}

; Test a case where it is the source rather than destination of LER that
; we need, but cannot convert the LER.
define float @f15(float %val, float %dummy) #0 {
; CHECK-LABEL: f15:
; CHECK: ler %f2, %f0
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ltebr %f1, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %ret = call float asm "blah $1", "=f,{f2}"(float %val) #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
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

; Test a case where it is the source rather than destination of LDR that
; we need, but cannot convert the LDR.
define double @f16(double %val, double %dummy) #0 {
; CHECK-LABEL: f16:
; CHECK: ldr %f2, %f0
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ltdbr %f1, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %ret = call double asm "blah $1", "=f,{f2}"(double %val) #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f64(
                                               double %val, double 0.0,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  call void asm sideeffect "blah", ""() #0
  br label %exit

exit:
  ret double %ret
}

; Repeat f2 with a comparison against -0.
define float @f17(float %a, float %b, ptr %dest) #0 {
; CHECK-LABEL: f17:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float -0.0,
                                               metadata !"olt",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Verify that we cannot omit the compare if there may be an intervening
; change to the exception flags.
define float @f18(float %a, float %b, ptr %dest) #0 {
; CHECK-LABEL: f18:
; CHECK: aebr %f0, %f2
; CHECK: ltebr %f1, %f0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.experimental.constrained.fadd.f32(
                        float %a, float %b,
                        metadata !"round.dynamic",
                        metadata !"fpexcept.strict") #0
  call void asm sideeffect "blah", ""() #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
                                               float %res, float 0.0,
                                               metadata !"oeq",
                                               metadata !"fpexcept.strict") #0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Verify that we cannot convert LER to LTEBR and omit the compare if
; there may be an intervening change to the exception flags.
define float @f19(float %dummy, float %val) #0 {
; CHECK-LABEL: f19:
; CHECK: ler %f0, %f2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f0
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: ltebr %f1, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %ret = call float asm sideeffect "blah $1", "=f,{f0}"(float %val) #0
  %cmp = call i1 @llvm.experimental.constrained.fcmp.f32(
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
declare i1 @llvm.experimental.constrained.fcmp.f32(float, float, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f64(double, double, metadata, metadata)
declare i1 @llvm.experimental.constrained.fcmp.f128(fp128, fp128, metadata, metadata)
