; RUN: llc -verify-machineinstrs -o - %s -mtriple=arm64-apple-ios7.0 -aarch64-atomic-cfg-tidy=0 | FileCheck %s
; RUN: llc -verify-machineinstrs < %s -mtriple=aarch64-none-linux-gnu -mattr=-fp-armv8 -aarch64-atomic-cfg-tidy=0 | FileCheck --check-prefix=CHECK-NOFP %s

define i32 @test_select_i32(i1 %bit, i32 %a, i32 %b) {
; CHECK-LABEL: test_select_i32:
  %val = select i1 %bit, i32 %a, i32 %b
; CHECK: tst w0, #0x1
; CHECK-NEXT: csel w0, w1, w2, ne

  ret i32 %val
}

define i64 @test_select_i64(i1 %bit, i64 %a, i64 %b) {
; CHECK-LABEL: test_select_i64:
  %val = select i1 %bit, i64 %a, i64 %b
; CHECK: tst w0, #0x1
; CHECK-NEXT: csel x0, x1, x2, ne

  ret i64 %val
}

define float @test_select_float(i1 %bit, float %a, float %b) {
; CHECK-LABEL: test_select_float:
  %val = select i1 %bit, float %a, float %b
; CHECK: tst w0, #0x1
; CHECK-NEXT: fcsel s0, s0, s1, ne
; CHECK-NOFP-NOT: fcsel
  ret float %val
}

define double @test_select_double(i1 %bit, double %a, double %b) {
; CHECK-LABEL: test_select_double:
  %val = select i1 %bit, double %a, double %b
; CHECK: tst w0, #0x1
; CHECK-NEXT: fcsel d0, d0, d1, ne
; CHECK-NOFP-NOT: fcsel

  ret double %val
}

define i32 @test_brcond(i1 %bit) {
; CHECK-LABEL: test_brcond:
  br i1 %bit, label %true, label %false
; CHECK: tbz {{w[0-9]+}}, #0, {{.?LBB}}

true:
  ret i32 0
false:
  ret i32 42
}

define i1 @test_setcc_float(float %lhs, float %rhs) {
; CHECK: test_setcc_float
  %val = fcmp oeq float %lhs, %rhs
; CHECK: fcmp s0, s1
; CHECK: cset w0, eq
; CHECK-NOFP-NOT: fcmp
  ret i1 %val
}

define i1 @test_setcc_double(double %lhs, double %rhs) {
; CHECK: test_setcc_double
  %val = fcmp oeq double %lhs, %rhs
; CHECK: fcmp d0, d1
; CHECK: cset w0, eq
; CHECK-NOFP-NOT: fcmp
  ret i1 %val
}

define i1 @test_setcc_i32(i32 %lhs, i32 %rhs) {
; CHECK: test_setcc_i32
  %val = icmp ugt i32 %lhs, %rhs
; CHECK: cmp w0, w1
; CHECK: cset w0, hi
  ret i1 %val
}

define i1 @test_setcc_i64(i64 %lhs, i64 %rhs) {
; CHECK: test_setcc_i64
  %val = icmp ne i64 %lhs, %rhs
; CHECK: cmp x0, x1
; CHECK: cset w0, ne
  ret i1 %val
}
