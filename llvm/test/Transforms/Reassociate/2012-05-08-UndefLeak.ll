; RUN: opt < %s -reassociate -S | FileCheck %s
; PR12169
; PR12764
; XFAIL: *
; Transform disabled until PR13021 is fixed.

define i64 @f(i64 %x0) {
; CHECK-LABEL: @f(
; CHECK-NEXT: mul i64 %x0, 208
; CHECK-NEXT: add i64 %{{.*}}, 1617
; CHECK-NEXT: ret i64
  %t0 = add i64 %x0, 1
  %t1 = add i64 %x0, 2
  %t2 = add i64 %x0, 3
  %t3 = add i64 %x0, 4
  %t4 = add i64 %x0, 5
  %t5 = add i64 %x0, 6
  %t6 = add i64 %x0, 7
  %t7 = add i64 %x0, 8
  %t8 = add i64 %x0, 9
  %t9 = add i64 %x0, 10
  %t10 = add i64 %x0, 11
  %t11 = add i64 %x0, 12
  %t12 = add i64 %x0, 13
  %t13 = add i64 %x0, 14
  %t14 = add i64 %x0, 15
  %t15 = add i64 %x0, 16
  %t16 = add i64 %x0, 17
  %t17 = add i64 %x0, 18
  %t18 = add i64 %t17, %t0
  %t19 = add i64 %t18, %t1
  %t20 = add i64 %t19, %t2
  %t21 = add i64 %t20, %t3
  %t22 = add i64 %t21, %t4
  %t23 = add i64 %t22, %t5
  %t24 = add i64 %t23, %t6
  %t25 = add i64 %t24, %t7
  %t26 = add i64 %t25, %t8
  %t27 = add i64 %t26, %t9
  %t28 = add i64 %t27, %t10
  %t29 = add i64 %t28, %t11
  %t30 = add i64 %t29, %t12
  %t31 = add i64 %t30, %t13
  %t32 = add i64 %t31, %t14
  %t33 = add i64 %t32, %t15
  %t34 = add i64 %t33, %t16
  %t35 = add i64 %t34, %x0
  %t36 = add i64 %t0, %t1
  %t37 = add i64 %t36, %t2
  %t38 = add i64 %t37, %t3
  %t39 = add i64 %t38, %t4
  %t40 = add i64 %t39, %t5
  %t41 = add i64 %t40, %t6
  %t42 = add i64 %t41, %t7
  %t43 = add i64 %t42, %t8
  %t44 = add i64 %t43, %t9
  %t45 = add i64 %t44, %t10
  %t46 = add i64 %t45, %t11
  %t47 = add i64 %t46, %t12
  %t48 = add i64 %t47, %t13
  %t49 = add i64 %t48, %t14
  %t50 = add i64 %t49, %t15
  %t51 = add i64 %t50, %t16
  %t52 = add i64 %t51, %t17
  %t53 = add i64 %t52, %t18
  %t54 = add i64 %t53, %t19
  %t55 = add i64 %t54, %t20
  %t56 = add i64 %t55, %t21
  %t57 = add i64 %t56, %t22
  %t58 = add i64 %t57, %t23
  %t59 = add i64 %t58, %t24
  %t60 = add i64 %t59, %t25
  %t61 = add i64 %t60, %t26
  %t62 = add i64 %t61, %t27
  %t63 = add i64 %t62, %t28
  %t64 = add i64 %t63, %t29
  %t65 = add i64 %t64, %t30
  %t66 = add i64 %t65, %t31
  %t67 = add i64 %t66, %t32
  %t68 = add i64 %t67, %t33
  %t69 = add i64 %t68, %t34
  %t70 = add i64 %t69, %t35
  %t71 = add i64 %t70, %x0
  ret i64 %t71
}
