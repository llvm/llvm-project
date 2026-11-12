; Test all condition-code masks that are relevant for floating-point
; comparisons.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

define void @f1(ptr %src, float %target) {
; CHECK-LABEL: f1:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: je .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp oeq float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f2(ptr %src, float %target) {
; CHECK-LABEL: f2:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jlh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp one float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f3(ptr %src, float %target) {
; CHECK-LABEL: f3:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jle .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp ole float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f4(ptr %src, float %target) {
; CHECK-LABEL: f4:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jl .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp olt float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f5(ptr %src, float %target) {
; CHECK-LABEL: f5:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp ogt float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f6(ptr %src, float %target) {
; CHECK-LABEL: f6:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jhe .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp oge float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f7(ptr %src, float %target) {
; CHECK-LABEL: f7:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jnlh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp ueq float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f8(ptr %src, float %target) {
; CHECK-LABEL: f8:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jne .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp une float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f9(ptr %src, float %target) {
; CHECK-LABEL: f9:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jnh .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp ule float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f10(ptr %src, float %target) {
; CHECK-LABEL: f10:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jnhe .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp ult float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f11(ptr %src, float %target) {
; CHECK-LABEL: f11:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jnle .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp ugt float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

define void @f12(ptr %src, float %target) {
; CHECK-LABEL: f12:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jnl .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp uge float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; "jno" == "jump if no overflow", which corresponds to "jump if ordered"
; rather than "jump if not ordered" after a floating-point comparison.
define void @f13(ptr %src, float %target) {
; CHECK-LABEL: f13:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jno .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp ord float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}

; "jo" == "jump if overflow", which corresponds to "jump if not ordered"
; rather than "jump if ordered" after a floating-point comparison.
define void @f14(ptr %src, float %target) {
; CHECK-LABEL: f14:
; CHECK: .cfi_startproc
; CHECK: .L[[LABEL:.*]]:
; CHECK: ceb %f0, 0(%r2)
; CHECK-NEXT: jo .L[[LABEL]]
  br label %loop
loop:
  %val = load volatile float, ptr %src
  %cond = fcmp uno float %target, %val
  br i1 %cond, label %loop, label %exit
exit:
  ret void
}
