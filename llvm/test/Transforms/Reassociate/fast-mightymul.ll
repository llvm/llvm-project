; RUN: opt < %s -passes=reassociate -disable-output
; PR13021

define float @test2(float %x) {
  %t0 = fmul fast float %x, %x
  %t1 = fmul fast float %t0, %t0
  %t2 = fmul fast float %t1, %t1
  %t3 = fmul fast float %t2, %t2
  %t4 = fmul fast float %t3, %t3
  %t5 = fmul fast float %t4, %t4
  %t6 = fmul fast float %t5, %t5
  %t7 = fmul fast float %t6, %t6
  %t8 = fmul fast float %t7, %t7
  %t9 = fmul fast float %t8, %t8
  %t10 = fmul fast float %t9, %t9
  %t11 = fmul fast float %t10, %t10
  %t12 = fmul fast float %t11, %t11
  %t13 = fmul fast float %t12, %t12
  %t14 = fmul fast float %t13, %t13
  %t15 = fmul fast float %t14, %t14
  %t16 = fmul fast float %t15, %t15
  %t17 = fmul fast float %t16, %t16
  %t18 = fmul fast float %t17, %t17
  %t19 = fmul fast float %t18, %t18
  %t20 = fmul fast float %t19, %t19
  %t21 = fmul fast float %t20, %t20
  %t22 = fmul fast float %t21, %t21
  %t23 = fmul fast float %t22, %t22
  %t24 = fmul fast float %t23, %t23
  %t25 = fmul fast float %t24, %t24
  %t26 = fmul fast float %t25, %t25
  %t27 = fmul fast float %t26, %t26
  %t28 = fmul fast float %t27, %t27
  ret float %t28
}
