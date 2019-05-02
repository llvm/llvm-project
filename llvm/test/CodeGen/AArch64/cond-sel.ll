; RUN: llc -enable-machine-outliner=never -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mcpu=cyclone | FileCheck %s
; RUN: llc -enable-machine-outliner=never -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 | FileCheck --check-prefix=CHECK-NOFP %s

@var32 = global i32 0
@var64 = global i64 0

define void @test_csel(i32 %lhs32, i32 %rhs32, i64 %lhs64) minsize {
; CHECK-LABEL: test_csel:

  %tst1 = icmp ugt i32 %lhs32, %rhs32
  %val1 = select i1 %tst1, i32 42, i32 52
  store i32 %val1, i32* @var32
; CHECK-DAG: mov [[W52:w[0-9]+]], #{{52|0x34}}
; CHECK-DAG: mov [[W42:w[0-9]+]], #{{42|0x2a}}
; CHECK: csel {{w[0-9]+}}, [[W42]], [[W52]], hi

  %rhs64 = sext i32 %rhs32 to i64
  %tst2 = icmp sle i64 %lhs64, %rhs64
  %val2 = select i1 %tst2, i64 %lhs64, i64 %rhs64
  store i64 %val2, i64* @var64
; CHECK: sxtw [[EXT_RHS:x[0-9]+]], {{[wx]}}[[RHS:[0-9]+]]
; CHECK: cmp [[LHS:x[0-9]+]], w[[RHS]], sxtw
; CHECK: csel {{x[0-9]+}}, [[LHS]], [[EXT_RHS]], le

  ret void
; CHECK: ret
}

define void @test_floatcsel(float %lhs32, float %rhs32, double %lhs64, double %rhs64) {
; CHECK-LABEL: test_floatcsel:

  %tst1 = fcmp one float %lhs32, %rhs32
; CHECK: fcmp {{s[0-9]+}}, {{s[0-9]+}}
; CHECK-NOFP-NOT: fcmp
  %val1 = select i1 %tst1, i32 42, i32 52
  store i32 %val1, i32* @var32
; CHECK: mov [[W52:w[0-9]+]], #{{52|0x34}}
; CHECK: mov [[W42:w[0-9]+]], #{{42|0x2a}}
; CHECK: csel [[MAYBETRUE:w[0-9]+]], [[W42]], [[W52]], mi
; CHECK: csel {{w[0-9]+}}, [[W42]], [[MAYBETRUE]], gt


  %tst2 = fcmp ueq double %lhs64, %rhs64
; CHECK: fcmp {{d[0-9]+}}, {{d[0-9]+}}
; CHECK-NOFP-NOT: fcmp
  %val2 = select i1 %tst2, i64 9, i64 15
  store i64 %val2, i64* @var64
; CHECK: mov w[[CONST15:[0-9]+]], #15
; CHECK: mov {{[wx]}}[[CONST9:[0-9]+]], #{{9|0x9}}
; CHECK: csel [[MAYBETRUE:x[0-9]+]], x[[CONST9]], x[[CONST15]], eq
; CHECK: csel {{x[0-9]+}}, x[[CONST9]], [[MAYBETRUE]], vs

  ret void
; CHECK: ret
}


define void @test_csinc(i32 %lhs32, i32 %rhs32, i64 %lhs64) minsize {
; CHECK-LABEL: test_csinc:

; Note that commuting rhs and lhs in the select changes ugt to ule (i.e. hi to ls).
  %tst1 = icmp ugt i32 %lhs32, %rhs32
  %inc1 = add i32 %rhs32, 1
  %val1 = select i1 %tst1, i32 %inc1, i32 %lhs32
  store volatile i32 %val1, i32* @var32
; CHECK: cmp [[LHS:w[0-9]+]], [[RHS:w[0-9]+]]
; CHECK: csinc {{w[0-9]+}}, [[LHS]], [[RHS]], ls

  %rhs2 = add i32 %rhs32, 42
  %tst2 = icmp sle i32 %lhs32, %rhs2
  %inc2 = add i32 %rhs32, 1
  %val2 = select i1 %tst2, i32 %lhs32, i32 %inc2
  store volatile i32 %val2, i32* @var32
; CHECK: cmp [[LHS:w[0-9]+]], {{w[0-9]+}}
; CHECK: csinc {{w[0-9]+}}, [[LHS]], {{w[0-9]+}}, le

; Note that commuting rhs and lhs in the select changes ugt to ule (i.e. hi to ls).
  %rhs3 = sext i32 %rhs32 to i64
  %tst3 = icmp ugt i64 %lhs64, %rhs3
  %inc3 = add i64 %rhs3, 1
  %val3 = select i1 %tst3, i64 %inc3, i64 %lhs64
  store volatile i64 %val3, i64* @var64
; CHECK: cmp [[LHS:x[0-9]+]], {{w[0-9]+}}
; CHECK: csinc {{x[0-9]+}}, [[LHS]], {{x[0-9]+}}, ls

  %rhs4 = zext i32 %rhs32 to i64
  %tst4 = icmp sle i64 %lhs64, %rhs4
  %inc4 = add i64 %rhs4, 1
  %val4 = select i1 %tst4, i64 %lhs64, i64 %inc4
  store volatile i64 %val4, i64* @var64
; CHECK: cmp [[LHS:x[0-9]+]], {{w[0-9]+}}
; CHECK: csinc {{x[0-9]+}}, [[LHS]], {{x[0-9]+}}, le

  ret void
; CHECK: ret
}

define void @test_csinv(i32 %lhs32, i32 %rhs32, i64 %lhs64) minsize {
; CHECK-LABEL: test_csinv:

; Note that commuting rhs and lhs in the select changes ugt to ule (i.e. hi to ls).
  %tst1 = icmp ugt i32 %lhs32, %rhs32
  %inc1 = xor i32 -1, %rhs32
  %val1 = select i1 %tst1, i32 %inc1, i32 %lhs32
  store volatile i32 %val1, i32* @var32
; CHECK: cmp [[LHS:w[0-9]+]], [[RHS:w[0-9]+]]
; CHECK: csinv {{w[0-9]+}}, [[LHS]], [[RHS]], ls

  %rhs2 = add i32 %rhs32, 42
  %tst2 = icmp sle i32 %lhs32, %rhs2
  %inc2 = xor i32 -1, %rhs32
  %val2 = select i1 %tst2, i32 %lhs32, i32 %inc2
  store volatile i32 %val2, i32* @var32
; CHECK: cmp [[LHS:w[0-9]+]], {{w[0-9]+}}
; CHECK: csinv {{w[0-9]+}}, [[LHS]], {{w[0-9]+}}, le

; Note that commuting rhs and lhs in the select changes ugt to ule (i.e. hi to ls).
  %rhs3 = sext i32 %rhs32 to i64
  %tst3 = icmp ugt i64 %lhs64, %rhs3
  %inc3 = xor i64 -1, %rhs3
  %val3 = select i1 %tst3, i64 %inc3, i64 %lhs64
  store volatile i64 %val3, i64* @var64
; CHECK: cmp [[LHS:x[0-9]+]], {{w[0-9]+}}
; CHECK: csinv {{x[0-9]+}}, [[LHS]], {{x[0-9]+}}, ls

  %rhs4 = zext i32 %rhs32 to i64
  %tst4 = icmp sle i64 %lhs64, %rhs4
  %inc4 = xor i64 -1, %rhs4
  %val4 = select i1 %tst4, i64 %lhs64, i64 %inc4
  store volatile i64 %val4, i64* @var64
; CHECK: cmp [[LHS:x[0-9]+]], {{w[0-9]+}}
; CHECK: csinv {{x[0-9]+}}, [[LHS]], {{x[0-9]+}}, le

  ret void
; CHECK: ret
}

define void @test_csinv0(i32 %lhs32, i32 %rhs32, i64 %lhs64, i64 %rhs64) minsize {
; CHECK-LABEL: test_csinv0:

  %tst1 = icmp ugt i32 %lhs32, %rhs32
  %val1 = select i1 %tst1, i32 0, i32 -1
  store volatile i32 %val1, i32* @var32
; CHECK: cmp [[LHS:w[0-9]+]], [[RHS:w[0-9]+]]
; CHECK: csetm {{w[0-9]+}}, ls

  %rhs2 = add i32 %rhs32, 42
  %tst2 = icmp sle i32 %lhs32, %rhs2
  %val2 = select i1 %tst2, i32 -1, i32 %rhs2
  store volatile i32 %val2, i32* @var32
; CHECK: cmp [[LHS2:w[0-9]+]], [[RHS2:w[0-9]+]]
; CHECK: csinv {{w[0-9]+}}, [[RHS2]], wzr, gt

; Note that commuting rhs and lhs in the select changes ugt to ule (i.e. hi to ls).
  %rhs3 = mul i64 %rhs64, 19
  %tst3 = icmp ugt i64 %lhs64, %rhs3
  %val3 = select i1 %tst3, i64 %rhs3, i64 -1
  store volatile i64 %val3, i64* @var64
; CHECK: cmp [[LHS3:x[0-9]+]], [[RHS3:x[0-9]+]]
; CHECK: csinv {{x[0-9]+}}, [[RHS3]], xzr, hi

  ret void
; CHECK: ret
}

define void @test_csneg(i32 %lhs32, i32 %rhs32, i64 %lhs64) minsize {
; CHECK-LABEL: test_csneg:

; Note that commuting rhs and lhs in the select changes ugt to ule (i.e. hi to ls).
  %tst1 = icmp ugt i32 %lhs32, %rhs32
  %inc1 = sub i32 0, %rhs32
  %val1 = select i1 %tst1, i32 %inc1, i32 %lhs32
  store volatile i32 %val1, i32* @var32
; CHECK: cmp [[LHS:w[0-9]+]], [[RHS:w[0-9]+]]
; CHECK: csneg {{w[0-9]+}}, [[LHS]], [[RHS]], ls

  %rhs2 = add i32 %rhs32, 42
  %tst2 = icmp sle i32 %lhs32, %rhs2
  %inc2 = sub i32 0, %rhs32
  %val2 = select i1 %tst2, i32 %lhs32, i32 %inc2
  store volatile i32 %val2, i32* @var32
; CHECK: cmp [[LHS:w[0-9]+]], {{w[0-9]+}}
; CHECK: csneg {{w[0-9]+}}, [[LHS]], {{w[0-9]+}}, le

; Note that commuting rhs and lhs in the select changes ugt to ule (i.e. hi to ls).
  %rhs3 = sext i32 %rhs32 to i64
  %tst3 = icmp ugt i64 %lhs64, %rhs3
  %inc3 = sub i64 0, %rhs3
  %val3 = select i1 %tst3, i64 %inc3, i64 %lhs64
  store volatile i64 %val3, i64* @var64
; CHECK: cmp [[LHS:x[0-9]+]], {{w[0-9]+}}
; CHECK: csneg {{x[0-9]+}}, [[LHS]], {{x[0-9]+}}, ls

  %rhs4 = zext i32 %rhs32 to i64
  %tst4 = icmp sle i64 %lhs64, %rhs4
  %inc4 = sub i64 0, %rhs4
  %val4 = select i1 %tst4, i64 %lhs64, i64 %inc4
  store volatile i64 %val4, i64* @var64
; CHECK: cmp [[LHS:x[0-9]+]], {{w[0-9]+}}
; CHECK: csneg {{x[0-9]+}}, [[LHS]], {{x[0-9]+}}, le

  ret void
; CHECK: ret
}

define void @test_cset(i32 %lhs, i32 %rhs, i64 %lhs64) {
; CHECK-LABEL: test_cset:

; N.b. code is not optimal here (32-bit csinc would be better) but
; incoming DAG is too complex
  %tst1 = icmp eq i32 %lhs, %rhs
  %val1 = zext i1 %tst1 to i32
  store i32 %val1, i32* @var32
; CHECK: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: cset {{w[0-9]+}}, eq

  %rhs64 = sext i32 %rhs to i64
  %tst2 = icmp ule i64 %lhs64, %rhs64
  %val2 = zext i1 %tst2 to i64
  store i64 %val2, i64* @var64
; CHECK: cset {{w[0-9]+}}, ls

  ret void
; CHECK: ret
}

define void @test_csetm(i32 %lhs, i32 %rhs, i64 %lhs64) {
; CHECK-LABEL: test_csetm:

  %tst1 = icmp eq i32 %lhs, %rhs
  %val1 = sext i1 %tst1 to i32
  store i32 %val1, i32* @var32
; CHECK: cmp {{w[0-9]+}}, {{w[0-9]+}}
; CHECK: csetm {{w[0-9]+}}, eq

  %rhs64 = sext i32 %rhs to i64
  %tst2 = icmp ule i64 %lhs64, %rhs64
  %val2 = sext i1 %tst2 to i64
  store i64 %val2, i64* @var64
; CHECK: csetm {{x[0-9]+}}, ls

  ret void
; CHECK: ret
}

define <1 x i1> @test_wide_comparison(i32 %in) {
; CHECK-LABEL: test_wide_comparison:
; CHECK: cmp w0, #1234
; CHECK: cset

  %tmp = icmp sgt i32 %in, 1234
  %res = select i1 %tmp, <1 x i1> <i1 1>, <1 x i1> zeroinitializer
  ret <1 x i1> %res
}

define i32 @test_select_undef() {
; CHECK-LABEL: test_select_undef:
; CHECK: ret
  %res = select i1 undef, i32 0, i32 42
  ret i32 %res
}
