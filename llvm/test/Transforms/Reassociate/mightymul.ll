; RUN: opt < %s -reassociate -disable-output
; PR13021

define i32 @test1(i32 %x) {
  %t0 = mul i32 %x, %x
  %t1 = mul i32 %t0, %t0
  %t2 = mul i32 %t1, %t1
  %t3 = mul i32 %t2, %t2
  %t4 = mul i32 %t3, %t3
  %t5 = mul i32 %t4, %t4
  %t6 = mul i32 %t5, %t5
  %t7 = mul i32 %t6, %t6
  %t8 = mul i32 %t7, %t7
  %t9 = mul i32 %t8, %t8
  %t10 = mul i32 %t9, %t9
  %t11 = mul i32 %t10, %t10
  %t12 = mul i32 %t11, %t11
  %t13 = mul i32 %t12, %t12
  %t14 = mul i32 %t13, %t13
  %t15 = mul i32 %t14, %t14
  %t16 = mul i32 %t15, %t15
  %t17 = mul i32 %t16, %t16
  %t18 = mul i32 %t17, %t17
  %t19 = mul i32 %t18, %t18
  %t20 = mul i32 %t19, %t19
  %t21 = mul i32 %t20, %t20
  %t22 = mul i32 %t21, %t21
  %t23 = mul i32 %t22, %t22
  %t24 = mul i32 %t23, %t23
  %t25 = mul i32 %t24, %t24
  %t26 = mul i32 %t25, %t25
  %t27 = mul i32 %t26, %t26
  %t28 = mul i32 %t27, %t27
  ret i32 %t28
}
