; RUN: llc -march=hexagon < %s | FileCheck %s

@g0 = common global i8 0, align 1
@g1 = common global i8 0, align 1
@g2 = common global i16 0, align 2
@g3 = common global i16 0, align 2
@g4 = common global i32 0, align 4
@g5 = common global i32 0, align 4

; CHECK-LABEL: f0:
; CHECK: memb(r{{[0-9]+}}+#0) += #1
define void @f0() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = add i8 %v0, 1
  store i8 %v1, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f1:
; CHECK: memb(r{{[0-9]+}}+#0) -= #1
define void @f1() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = add i8 %v0, -1
  store i8 %v1, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f2:
; CHECK: memb(r{{[0-9]+}}+#0) += #5
define void @f2() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, 5
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f3:
; CHECK: memb(r{{[0-9]+}}+#0) -= #5
define void @f3() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, 251
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f4:
; CHECK: memb(r{{[0-9]+}}+#0) -= #5
define void @f4() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, 251
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f5:
; CHECK: memb(r{{[0-9]+}}+#0) += #5
define void @f5() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, 5
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f6:
; CHECK: memb(r{{[0-9]+}}+#0) += r{{[0-9]+}}
define void @f6(i8 zeroext %a0) #0 {
b0:
  %v0 = zext i8 %a0 to i32
  %v1 = load i8, ptr @g0, align 1, !tbaa !0
  %v2 = zext i8 %v1 to i32
  %v3 = add nsw i32 %v2, %v0
  %v4 = trunc i32 %v3 to i8
  store i8 %v4, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f7:
; CHECK: memb(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
define void @f7(i8 zeroext %a0) #0 {
b0:
  %v0 = zext i8 %a0 to i32
  %v1 = load i8, ptr @g0, align 1, !tbaa !0
  %v2 = zext i8 %v1 to i32
  %v3 = sub nsw i32 %v2, %v0
  %v4 = trunc i32 %v3 to i8
  store i8 %v4, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f8:
; CHECK: memb(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
define void @f8(i8 zeroext %a0) #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = or i8 %v0, %a0
  store i8 %v1, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f9:
; CHECK: memb(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
define void @f9(i8 zeroext %a0) #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = and i8 %v0, %a0
  store i8 %v1, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f10:
; CHECK: memb(r{{[0-9]+}}+#0) = clrbit(#5)
define void @f10() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = and i32 %v1, 223
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f11:
; CHECK: memb(r{{[0-9]+}}+#0) = setbit(#7)
define void @f11() #0 {
b0:
  %v0 = load i8, ptr @g0, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = or i32 %v1, 128
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g0, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f12:
; CHECK: memb(r{{[0-9]+}}+#0) += #1
define void @f12() #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = add i8 %v0, 1
  store i8 %v1, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f13:
; CHECK: memb(r{{[0-9]+}}+#0) -= #1
define void @f13() #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = add i8 %v0, -1
  store i8 %v1, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f14:
; CHECK: memb(r{{[0-9]+}}+#0) += #5
define void @f14() #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, 5
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f15:
; CHECK: memb(r{{[0-9]+}}+#0) -= #5
define void @f15() #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, 251
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f16:
; CHECK: memb(r{{[0-9]+}}+#0) -= #5
define void @f16() #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, 251
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f17:
; CHECK: memb(r{{[0-9]+}}+#0) += #5
define void @f17() #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = add nsw i32 %v1, 5
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f18:
; CHECK: memb(r{{[0-9]+}}+#0) += r{{[0-9]+}}
define void @f18(i8 signext %a0) #0 {
b0:
  %v0 = zext i8 %a0 to i32
  %v1 = load i8, ptr @g1, align 1, !tbaa !0
  %v2 = zext i8 %v1 to i32
  %v3 = add nsw i32 %v2, %v0
  %v4 = trunc i32 %v3 to i8
  store i8 %v4, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f19:
; CHECK: memb(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
define void @f19(i8 signext %a0) #0 {
b0:
  %v0 = zext i8 %a0 to i32
  %v1 = load i8, ptr @g1, align 1, !tbaa !0
  %v2 = zext i8 %v1 to i32
  %v3 = sub nsw i32 %v2, %v0
  %v4 = trunc i32 %v3 to i8
  store i8 %v4, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f20:
; CHECK: memb(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
define void @f20(i8 signext %a0) #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = or i8 %v0, %a0
  store i8 %v1, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f21:
; CHECK: memb(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
define void @f21(i8 signext %a0) #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = and i8 %v0, %a0
  store i8 %v1, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f22:
; CHECK: memb(r{{[0-9]+}}+#0) = clrbit(#5)
define void @f22() #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = and i32 %v1, 223
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f23:
; CHECK: memb(r{{[0-9]+}}+#0) = setbit(#7)
define void @f23() #0 {
b0:
  %v0 = load i8, ptr @g1, align 1, !tbaa !0
  %v1 = zext i8 %v0 to i32
  %v2 = or i32 %v1, 128
  %v3 = trunc i32 %v2 to i8
  store i8 %v3, ptr @g1, align 1, !tbaa !0
  ret void
}

; CHECK-LABEL: f24:
; CHECK: memh(r{{[0-9]+}}+#0) += #1
define void @f24() #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = add i16 %v0, 1
  store i16 %v1, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f25:
; CHECK: memh(r{{[0-9]+}}+#0) -= #1
define void @f25() #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = add i16 %v0, -1
  store i16 %v1, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f26:
; CHECK: memh(r{{[0-9]+}}+#0) += #5
define void @f26() #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = add nsw i32 %v1, 5
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f27:
; CHECK: memh(r{{[0-9]+}}+#0) -= #5
define void @f27() #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = add nsw i32 %v1, 65531
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f28:
; CHECK: memh(r{{[0-9]+}}+#0) -= #5
define void @f28() #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = add nsw i32 %v1, 65531
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f29:
; CHECK: memh(r{{[0-9]+}}+#0) += #5
define void @f29() #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = add nsw i32 %v1, 5
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f30:
; CHECK: memh(r{{[0-9]+}}+#0) += r{{[0-9]+}}
define void @f30(i16 zeroext %a0) #0 {
b0:
  %v0 = zext i16 %a0 to i32
  %v1 = load i16, ptr @g2, align 2, !tbaa !3
  %v2 = zext i16 %v1 to i32
  %v3 = add nsw i32 %v2, %v0
  %v4 = trunc i32 %v3 to i16
  store i16 %v4, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f31:
; CHECK: memh(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
define void @f31(i16 zeroext %a0) #0 {
b0:
  %v0 = zext i16 %a0 to i32
  %v1 = load i16, ptr @g2, align 2, !tbaa !3
  %v2 = zext i16 %v1 to i32
  %v3 = sub nsw i32 %v2, %v0
  %v4 = trunc i32 %v3 to i16
  store i16 %v4, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f32:
; CHECK: memh(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
define void @f32(i16 zeroext %a0) #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = or i16 %v0, %a0
  store i16 %v1, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f33:
; CHECK: memh(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
define void @f33(i16 zeroext %a0) #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = and i16 %v0, %a0
  store i16 %v1, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f34:
; CHECK: memh(r{{[0-9]+}}+#0) = clrbit(#5)
define void @f34() #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = and i32 %v1, 65503
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f35:
; CHECK: memh(r{{[0-9]+}}+#0) = setbit(#7)
define void @f35() #0 {
b0:
  %v0 = load i16, ptr @g2, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = or i32 %v1, 128
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g2, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f36:
; CHECK: memh(r{{[0-9]+}}+#0) += #1
define void @f36() #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = add i16 %v0, 1
  store i16 %v1, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f37:
; CHECK: memh(r{{[0-9]+}}+#0) -= #1
define void @f37() #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = add i16 %v0, -1
  store i16 %v1, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f38:
; CHECK: memh(r{{[0-9]+}}+#0) += #5
define void @f38() #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = add nsw i32 %v1, 5
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f39:
; CHECK: memh(r{{[0-9]+}}+#0) -= #5
define void @f39() #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = add nsw i32 %v1, 65531
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f40:
; CHECK: memh(r{{[0-9]+}}+#0) -= #5
define void @f40() #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = add nsw i32 %v1, 65531
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f41
; CHECK: memh(r{{[0-9]+}}+#0) += #5
define void @f41() #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = add nsw i32 %v1, 5
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f42
; CHECK: memh(r{{[0-9]+}}+#0) += r{{[0-9]+}}
define void @f42(i16 signext %a0) #0 {
b0:
  %v0 = zext i16 %a0 to i32
  %v1 = load i16, ptr @g3, align 2, !tbaa !3
  %v2 = zext i16 %v1 to i32
  %v3 = add nsw i32 %v2, %v0
  %v4 = trunc i32 %v3 to i16
  store i16 %v4, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f43
; CHECK: memh(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
define void @f43(i16 signext %a0) #0 {
b0:
  %v0 = zext i16 %a0 to i32
  %v1 = load i16, ptr @g3, align 2, !tbaa !3
  %v2 = zext i16 %v1 to i32
  %v3 = sub nsw i32 %v2, %v0
  %v4 = trunc i32 %v3 to i16
  store i16 %v4, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f44
; CHECK: memh(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
define void @f44(i16 signext %a0) #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = or i16 %v0, %a0
  store i16 %v1, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f45
; CHECK: memh(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
define void @f45(i16 signext %a0) #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = and i16 %v0, %a0
  store i16 %v1, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f46
; CHECK: memh(r{{[0-9]+}}+#0) = clrbit(#5)
define void @f46() #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = and i32 %v1, 65503
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f47
; CHECK: memh(r{{[0-9]+}}+#0) = setbit(#7)
define void @f47() #0 {
b0:
  %v0 = load i16, ptr @g3, align 2, !tbaa !3
  %v1 = zext i16 %v0 to i32
  %v2 = or i32 %v1, 128
  %v3 = trunc i32 %v2 to i16
  store i16 %v3, ptr @g3, align 2, !tbaa !3
  ret void
}

; CHECK-LABEL: f48
; CHECK: memw(r{{[0-9]+}}+#0) += #1
define void @f48() #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = add nsw i32 %v0, 1
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f49
; CHECK: memw(r{{[0-9]+}}+#0) -= #1
define void @f49() #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = add nsw i32 %v0, -1
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f50
; CHECK: memw(r{{[0-9]+}}+#0) += #5
define void @f50() #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = add nsw i32 %v0, 5
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f51
; CHECK: memw(r{{[0-9]+}}+#0) -= #5
define void @f51() #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = add nsw i32 %v0, -5
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f52
; CHECK: memw(r{{[0-9]+}}+#0) -= #5
define void @f52() #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = add nsw i32 %v0, -5
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f53
; CHECK: memw(r{{[0-9]+}}+#0) += #5
define void @f53() #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = add nsw i32 %v0, 5
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f54
; CHECK: memw(r{{[0-9]+}}+#0) += r{{[0-9]+}}
define void @f54(i32 %a0) #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = add i32 %v0, %a0
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f55
; CHECK: memw(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
define void @f55(i32 %a0) #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = sub i32 %v0, %a0
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f56
; CHECK: memw(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
define void @f56(i32 %a0) #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = or i32 %v0, %a0
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f57
; CHECK: memw(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
define void @f57(i32 %a0) #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = and i32 %v0, %a0
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f58
; CHECK: memw(r{{[0-9]+}}+#0) = clrbit(#5)
define void @f58() #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = and i32 %v0, -33
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f59
; CHECK: memw(r{{[0-9]+}}+#0) = setbit(#7)
define void @f59() #0 {
b0:
  %v0 = load i32, ptr @g4, align 4, !tbaa !5
  %v1 = or i32 %v0, 128
  store i32 %v1, ptr @g4, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f60
; CHECK: memw(r{{[0-9]+}}+#0) += #1
define void @f60() #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = add i32 %v0, 1
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f61
; CHECK: memw(r{{[0-9]+}}+#0) -= #1
define void @f61() #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = add i32 %v0, -1
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f62
; CHECK: memw(r{{[0-9]+}}+#0) += #5
define void @f62() #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = add i32 %v0, 5
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f63
; CHECK: memw(r{{[0-9]+}}+#0) -= #5
define void @f63() #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = add i32 %v0, -5
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f64
; CHECK: memw(r{{[0-9]+}}+#0) -= #5
define void @f64() #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = add i32 %v0, -5
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f65
; CHECK: memw(r{{[0-9]+}}+#0) += #5
define void @f65() #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = add i32 %v0, 5
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f66:
; CHECK: memw(r{{[0-9]+}}+#0) += r{{[0-9]+}}
define void @f66(i32 %a0) #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = add i32 %v0, %a0
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f67:
; CHECK: memw(r{{[0-9]+}}+#0) -= r{{[0-9]+}}
define void @f67(i32 %a0) #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = sub i32 %v0, %a0
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f68:
; CHECK: memw(r{{[0-9]+}}+#0) |= r{{[0-9]+}}
define void @f68(i32 %a0) #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = or i32 %v0, %a0
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f69:
; CHECK: memw(r{{[0-9]+}}+#0) &= r{{[0-9]+}}
define void @f69(i32 %a0) #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = and i32 %v0, %a0
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f70:
; CHECK: memw(r{{[0-9]+}}+#0) = clrbit(#5)
define void @f70() #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = and i32 %v0, -33
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

; CHECK-LABEL: f71:
; CHECK: memw(r{{[0-9]+}}+#0) = setbit(#7)
define void @f71() #0 {
b0:
  %v0 = load i32, ptr @g5, align 4, !tbaa !5
  %v1 = or i32 %v0, 128
  store i32 %v1, ptr @g5, align 4, !tbaa !5
  ret void
}

attributes #0 = { nounwind }

!0 = !{!1, !1, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!4, !4, i64 0}
!4 = !{!"short", !1}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !1}
