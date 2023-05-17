; RUN: llc < %s -mtriple=thumbv7-apple-ios %s -o - | FileCheck %s

@g0 = common global i32 0, align 4
@d0 = common global double 0.000000e+00, align 8
@f0 = common global float 0.000000e+00, align 4
@g1 = common global i32 0, align 4

declare i32 @llvm.arm.space(i32, i32)

; Check that the constant island pass moves the float constant pool entry inside
; the function.

; CHECK: .long 0x3f9e05ee @ float 1.23455596
; CHECK: {{.*}} %do.end

define i32 @testpadding(i32 %a) {
entry:
  %0 = load i32, ptr @g0, align 4
  %add = add nsw i32 %0, 12
  store i32 %add, ptr @g0, align 4
  %1 = load double, ptr @d0, align 8
  %add1 = fadd double %1, 0x3FF3C0B8ED46EACB
  store double %add1, ptr @d0, align 8
  %tmpcall11 = call i32 @llvm.arm.space(i32 28, i32 undef)
  call void @foo20(i32 191)
  %2 = load float, ptr @f0, align 4
  %add2 = fadd float %2, 0x3FF3C0BDC0000000
  store float %add2, ptr @f0, align 4
  br label %do.body

do.body:                                          ; preds = %do.body, %entry
  tail call void @foo20(i32 19)
  %3 = load i32, ptr @g1, align 4
  %tobool = icmp eq i32 %3, 0
  br i1 %tobool, label %do.end, label %do.body

do.end:                                           ; preds = %do.body
  %tmpcall111 = call i32 @llvm.arm.space(i32 954, i32 undef)
  ret i32 10
}

declare void @foo20(i32)
