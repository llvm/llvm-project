; Test that floating-point compares are omitted if CC already has the
; right value.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z10 -no-integrated-as | FileCheck %s

declare float @llvm.fabs.f32(float %f)

; Test addition followed by EQ, which can use the CC result of the addition.
define float @f1(float %a, float %b, ptr %dest) {
; CHECK-LABEL: f1:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %res = fadd float %a, %b
  %cmp = fcmp oeq float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; ...and again with LT.
define float @f2(float %a, float %b, ptr %dest) {
; CHECK-LABEL: f2:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = fadd float %a, %b
  %cmp = fcmp olt float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; ...and again with GT.
define float @f3(float %a, float %b, ptr %dest) {
; CHECK-LABEL: f3:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %res = fadd float %a, %b
  %cmp = fcmp ogt float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; ...and again with UEQ.
define float @f4(float %a, float %b, ptr %dest) {
; CHECK-LABEL: f4:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: bnlhr %r14
; CHECK: br %r14
entry:
  %res = fadd float %a, %b
  %cmp = fcmp ueq float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Subtraction also provides a zero-based CC value.
define float @f5(float %a, float %b, ptr %dest) {
; CHECK-LABEL: f5:
; CHECK: seb %f0, 0(%r2)
; CHECK-NEXT: bnher %r14
; CHECK: br %r14
entry:
  %cur = load float, ptr %dest
  %res = fsub float %a, %cur
  %cmp = fcmp ult float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD POSITIVE.
define float @f6(float %dummy, float %a, ptr %dest) {
; CHECK-LABEL: f6:
; CHECK: lpebr %f0, %f2
; CHECK-NEXT: bhr %r14
; CHECK: br %r14
entry:
  %res = call float @llvm.fabs.f32(float %a)
  %cmp = fcmp ogt float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD NEGATIVE.
define float @f7(float %dummy, float %a, ptr %dest) {
; CHECK-LABEL: f7:
; CHECK: lnebr %f0, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %abs = call float @llvm.fabs.f32(float %a)
  %res = fneg float %abs
  %cmp = fcmp olt float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test the result of LOAD COMPLEMENT.
define float @f8(float %dummy, float %a, ptr %dest) {
; CHECK-LABEL: f8:
; CHECK: lcebr %f0, %f2
; CHECK-NEXT: bler %r14
; CHECK: br %r14
entry:
  %res = fneg float %a
  %cmp = fcmp ole float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Multiplication (for example) does not modify CC.
define float @f9(float %a, float %b, ptr %dest) {
; CHECK-LABEL: f9:
; CHECK: meebr %f0, %f2
; CHECK-NEXT: ltebr %f0, %f0
; CHECK-NEXT: blhr %r14
; CHECK: br %r14
entry:
  %res = fmul float %a, %b
  %cmp = fcmp one float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test a combination involving a CC-setting instruction followed by
; a non-CC-setting instruction.
define float @f10(float %a, float %b, float %c, ptr %dest) {
; CHECK-LABEL: f10:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: debr %f0, %f4
; CHECK-NEXT: ltebr %f0, %f0
; CHECK-NEXT: bner %r14
; CHECK: br %r14
entry:
  %add = fadd float %a, %b
  %res = fdiv float %add, %c
  %cmp = fcmp une float %res, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test a case where CC is set based on a different register from the
; compare input.
define float @f11(float %a, float %b, float %c, ptr %dest1, ptr %dest2) {
; CHECK-LABEL: f11:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: sebr %f4, %f0
; CHECK-DAG: ste %f4, 0(%r2)
; CHECK-DAG: ltebr %f0, %f0
; CHECK-NEXT: ber %r14
; CHECK: br %r14
entry:
  %add = fadd float %a, %b
  %sub = fsub float %c, %add
  store float %sub, ptr %dest1
  %cmp = fcmp oeq float %add, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %sub, ptr %dest2
  br label %exit

exit:
  ret float %add
}

; Test that LER gets converted to LTEBR where useful.
define float @f12(float %dummy, float %val, ptr %dest) {
; CHECK-LABEL: f12:
; CHECK: ltebr %f0, %f2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f0
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  call void asm sideeffect "blah $0", "{f0}"(float %val)
  %cmp = fcmp olt float %val, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %val, ptr %dest
  br label %exit

exit:
  ret float %val
}

; Test that LDR gets converted to LTDBR where useful.
define double @f13(double %dummy, double %val, ptr %dest) {
; CHECK-LABEL: f13:
; CHECK: ltdbr %f0, %f2
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f0
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  call void asm sideeffect "blah $0", "{f0}"(double %val)
  %cmp = fcmp olt double %val, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store double %val, ptr %dest
  br label %exit

exit:
  ret double %val
}

; Test that LXR gets converted to LTXBR where useful.
define void @f14(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f14:
; CHECK: ltxbr
; CHECK-NEXT: dxbr
; CHECK-NEXT: std
; CHECK-NEXT: std
; CHECK-NEXT: mxbr
; CHECK-NEXT: std
; CHECK-NEXT: std
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %val1 = load fp128, ptr %ptr1
  %val2 = load fp128, ptr %ptr2
  %div = fdiv fp128 %val1, %val2
  store fp128 %div, ptr %ptr1
  %mul = fmul fp128 %val1, %val2
  store fp128 %mul, ptr %ptr2
  %cmp = fcmp olt fp128 %val1, 0xL00000000000000000000000000000000
  br i1 %cmp, label %exit, label %store

store:
  call void asm sideeffect "blah", ""()
  br label %exit

exit:
  ret void
}

; Test a case where it is the source rather than destination of LER that
; we need.
define float @f15(float %val, float %dummy, ptr %dest) {
; CHECK-LABEL: f15:
; CHECK: ltebr %f2, %f0
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  call void asm sideeffect "blah $0", "{f2}"(float %val)
  %cmp = fcmp olt float %val, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %val, ptr %dest
  br label %exit

exit:
  ret float %val
}

; Test a case where it is the source rather than destination of LDR that
; we need.
define double @f16(double %val, double %dummy, ptr %dest) {
; CHECK-LABEL: f16:
; CHECK: ltdbr %f2, %f0
; CHECK-NEXT: #APP
; CHECK-NEXT: blah %f2
; CHECK-NEXT: #NO_APP
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  call void asm sideeffect "blah $0", "{f2}"(double %val)
  %cmp = fcmp olt double %val, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store double %val, ptr %dest
  br label %exit

exit:
  ret double %val
}

; Repeat f2 with a comparison against -0.
define float @f17(float %a, float %b, ptr %dest) {
; CHECK-LABEL: f17:
; CHECK: aebr %f0, %f2
; CHECK-NEXT: blr %r14
; CHECK: br %r14
entry:
  %res = fadd float %a, %b
  %cmp = fcmp olt float %res, -0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %b, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Test another form of f7 in which the condition is based on the unnegated
; result.  This is what InstCombine would produce.
define float @f18(float %dummy, float %a, ptr %dest) {
; CHECK-LABEL: f18:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lnebr %f0, %f2
; CHECK-NEXT:    blr %r14
; CHECK-NEXT:  .LBB17_1: # %store
; CHECK-NEXT:    ste %f0, 0(%r2)
; CHECK-NEXT:    br %r14
entry:
  %abs = call float @llvm.fabs.f32(float %a)
  %res = fneg float %abs
  %cmp = fcmp ogt float %abs, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, ptr %dest
  br label %exit

exit:
  ret float %res
}

; Similarly for f8.
define float @f19(float %dummy, float %a, ptr %dest) {
; CHECK-LABEL: f19:
; CHECK:       # %bb.0: # %entry
; CHECK-NEXT:    lcebr %f0, %f2
; CHECK-NEXT:    bler %r14
; CHECK-NEXT:  .LBB18_1: # %store
; CHECK-NEXT:    ste %f0, 0(%r2)
; CHECK-NEXT:    br %r14
entry:
  %res = fneg float %a
  %cmp = fcmp oge float %a, 0.0
  br i1 %cmp, label %exit, label %store

store:
  store float %res, ptr %dest
  br label %exit

exit:
  ret float %res
}
