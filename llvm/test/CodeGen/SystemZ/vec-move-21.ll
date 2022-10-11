; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=z13 | FileCheck %s

; Test that a replicate of a load gets folded to vlrep also in cases where
; the load has multiple users.

; CHECK-NOT: vrep


define double @fun(ptr %Vsrc, <2 x double> %T) {
entry:
  %Vld1 = load double, ptr %Vsrc
  %Vgep2 = getelementptr double, ptr %Vsrc, i64 1
  %Vld2 = load double, ptr %Vgep2
  %Vgep3 = getelementptr double, ptr %Vsrc, i64 2
  %Vld3 = load double, ptr %Vgep3
  %Vgep4 = getelementptr double, ptr %Vsrc, i64 3
  %Vld4 = load double, ptr %Vgep4
  %Vgep5 = getelementptr double, ptr %Vsrc, i64 4
  %Vld5 = load double, ptr %Vgep5
  %Vgep6 = getelementptr double, ptr %Vsrc, i64 5
  %Vld6 = load double, ptr %Vgep6

  %V19 = insertelement <2 x double> undef, double %Vld1, i32 0
  %V20 = shufflevector <2 x double> %V19, <2 x double> undef, <2 x i32> zeroinitializer
  %V21 = insertelement <2 x double> undef, double %Vld4, i32 0
  %V22 = insertelement <2 x double> %V21, double %Vld5, i32 1
  %V23 = fmul <2 x double> %V20, %V22
  %V24 = fadd <2 x double> %T, %V23
  %V25 = insertelement <2 x double> %V19, double %Vld2, i32 1
  %V26 = insertelement <2 x double> undef, double %Vld6, i32 0
  %V27 = insertelement <2 x double> %V26, double %Vld6, i32 1
  %V28 = fmul <2 x double> %V25, %V27
  %V29 = fadd <2 x double> %T, %V28
  %V30 = insertelement <2 x double> undef, double %Vld2, i32 0
  %V31 = shufflevector <2 x double> %V30, <2 x double> undef, <2 x i32> zeroinitializer
  %V32 = insertelement <2 x double> undef, double %Vld5, i32 0
  %V33 = insertelement <2 x double> %V32, double %Vld6, i32 1
  %V34 = fmul <2 x double> %V31, %V33
  %V35 = fadd <2 x double> %T, %V34
  %V36 = insertelement <2 x double> undef, double %Vld3, i32 0
  %V37 = shufflevector <2 x double> %V36, <2 x double> undef, <2 x i32> zeroinitializer
  %V38 = fmul <2 x double> %V37, %V22
  %V39 = fadd <2 x double> %T, %V38
  %Vmul37 = fmul double %Vld3, %Vld6
  %Vadd38 = fadd double %Vmul37, %Vmul37

  %VA0 = fadd <2 x double> %V24, %V29
  %VA1 = fadd <2 x double> %VA0, %V35
  %VA2 = fadd <2 x double> %VA1, %V39

  %VE0 = extractelement <2 x double> %VA2, i32 0
  %VS1 = fadd double %VE0, %Vadd38

  ret double %VS1
}
