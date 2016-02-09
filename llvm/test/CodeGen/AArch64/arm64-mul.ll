; RUN: llc < %s -march=arm64 | FileCheck %s

; rdar://9296808
; rdar://9349137

define i128 @t1(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t1:
; CHECK: mul {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: umulh {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  %tmp1 = zext i64 %a to i128
  %tmp2 = zext i64 %b to i128
  %tmp3 = mul i128 %tmp1, %tmp2
  ret i128 %tmp3
}

define i128 @t2(i64 %a, i64 %b) nounwind readnone ssp {
entry:
; CHECK-LABEL: t2:
; CHECK: mul {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
; CHECK: smulh {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  %tmp1 = sext i64 %a to i128
  %tmp2 = sext i64 %b to i128
  %tmp3 = mul i128 %tmp1, %tmp2
  ret i128 %tmp3
}

define i64 @t3(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t3:
; CHECK: umull {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = zext i32 %a to i64
  %tmp2 = zext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @t4(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t4:
; CHECK: smull {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = sext i32 %a to i64
  %tmp2 = sext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  ret i64 %tmp3
}

define i64 @t5(i32 %a, i32 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL: t5:
; CHECK: umaddl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  %tmp1 = zext i32 %a to i64
  %tmp2 = zext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  %tmp4 = add i64 %c, %tmp3
  ret i64 %tmp4
}

define i64 @t6(i32 %a, i32 %b, i64 %c) nounwind {
entry:
; CHECK-LABEL: t6:
; CHECK: smsubl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  %tmp1 = sext i32 %a to i64
  %tmp2 = sext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  %tmp4 = sub i64 %c, %tmp3
  ret i64 %tmp4
}

define i64 @t7(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t7:
; CHECK: umnegl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = zext i32 %a to i64
  %tmp2 = zext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  %tmp4 = sub i64 0, %tmp3
  ret i64 %tmp4
}

define i64 @t8(i32 %a, i32 %b) nounwind {
entry:
; CHECK-LABEL: t8:
; CHECK: smnegl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = sext i32 %a to i64
  %tmp2 = sext i32 %b to i64
  %tmp3 = mul i64 %tmp1, %tmp2
  %tmp4 = sub i64 0, %tmp3
  ret i64 %tmp4
}

define i64 @t9(i32 %a) nounwind {
entry:
; CHECK-LABEL: t9:
; CHECK: umull {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = zext i32 %a to i64
  %tmp2 = mul i64 %tmp1, 139968
  ret i64 %tmp2
}

; Check 64-bit multiplication is used for constants > 32 bits.
define i64 @t10(i32 %a) nounwind {
entry:
; CHECK-LABEL: t10:
; CHECK: mul {{x[0-9]+}}, {{x[0-9]+}}, {{x[0-9]+}}
  %tmp1 = sext i32 %a to i64
  %tmp2 = mul i64 %tmp1, 2147483650 ; = 2^31 + 2
  ret i64 %tmp2
}

; Check the sext_inreg case.
define i64 @t11(i64 %a) nounwind {
entry:
; CHECK-LABEL: t11:
; CHECK: smnegl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}
  %tmp1 = trunc i64 %a to i32
  %tmp2 = sext i32 %tmp1 to i64
  %tmp3 = mul i64 %tmp2, -2395238
  %tmp4 = sub i64 0, %tmp3
  ret i64 %tmp4
}

define i64 @t12(i64 %a, i64 %b) nounwind {
entry:
; CHECK-LABEL: t12:
; CHECK: smaddl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  %tmp1 = trunc i64 %a to i32
  %tmp2 = sext i32 %tmp1 to i64
  %tmp3 = mul i64 %tmp2, -34567890
  %tmp4 = add i64 %b, %tmp3
  ret i64 %tmp4
}

define i64 @t13(i32 %a, i64 %b) nounwind {
entry:
; CHECK-LABEL: t13:
; CHECK: umsubl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  %tmp1 = zext i32 %a to i64
  %tmp3 = mul i64 %tmp1, 12345678
  %tmp4 = sub i64 %b, %tmp3
  ret i64 %tmp4
}

define i64 @t14(i32 %a, i64 %b) nounwind {
entry:
; CHECK-LABEL: t14:
; CHECK: smsubl {{x[0-9]+}}, {{w[0-9]+}}, {{w[0-9]+}}, {{x[0-9]+}}
  %tmp1 = sext i32 %a to i64
  %tmp3 = mul i64 %tmp1, -12345678
  %tmp4 = sub i64 %b, %tmp3
  ret i64 %tmp4
}
