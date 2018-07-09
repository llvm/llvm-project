; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -verify-machineinstrs < %s | FileCheck %s

@a_qp = common global fp128 0xL00000000000000000000000000000000, align 16
@b_qp = common global fp128 0xL00000000000000000000000000000000, align 16

; Function Attrs: noinline nounwind optnone
define signext i32 @greater_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp ogt fp128 %0, %1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: greater_qp
; CHECK: xscmpuqp
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, 1
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @less_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp olt fp128 %0, %1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: less_qp
; CHECK: xscmpuqp
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, 0
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @greater_eq_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp oge fp128 %0, %1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: greater_eq_qp
; CHECK: xscmpuqp
; CHECK: cror [[REG:[0-9]+]], {{[0-9]+}}, 0
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @less_eq_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp ole fp128 %0, %1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: less_eq_qp
; CHECK: xscmpuqp
; CHECK: cror [[REG:[0-9]+]], {{[0-9]+}}, 1
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @equal_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp oeq fp128 %0, %1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: equal_qp
; CHECK: xscmpuqp
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, 2
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @not_greater_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp ogt fp128 %0, %1
  %lnot = xor i1 %cmp, true
  %lnot.ext = zext i1 %lnot to i32
  ret i32 %lnot.ext
; CHECK-LABEL: not_greater_qp
; CHECK: xscmpuqp
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, 1
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @not_less_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp olt fp128 %0, %1
  %lnot = xor i1 %cmp, true
  %lnot.ext = zext i1 %lnot to i32
  ret i32 %lnot.ext
; CHECK-LABEL: not_less_qp
; CHECK: xscmpuqp
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, 0
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @not_greater_eq_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp oge fp128 %0, %1
  %lnot = xor i1 %cmp, true
  %lnot.ext = zext i1 %lnot to i32
  ret i32 %lnot.ext
; CHECK-LABEL: not_greater_eq_qp
; CHECK: xscmpuqp
; CHECK: crnor [[REG:[0-9]+]], 0, {{[0-9]+}}
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @not_less_eq_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp ole fp128 %0, %1
  %lnot = xor i1 %cmp, true
  %lnot.ext = zext i1 %lnot to i32
  ret i32 %lnot.ext
; CHECK-LABEL: not_less_eq_qp
; CHECK: xscmpuqp
; CHECK: crnor [[REG:[0-9]+]], 1, {{[0-9]+}}
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, [[REG]]
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define signext i32 @not_equal_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp une fp128 %0, %1
  %conv = zext i1 %cmp to i32
  ret i32 %conv
; CHECK-LABEL: not_equal_qp
; CHECK: xscmpuqp
; CHECK: isel {{[0-9]+}}, {{[0-9]+}}, {{[0-9]+}}, 2
; CHECK: blr
}

; Function Attrs: norecurse nounwind readonly
define fp128 @greater_sel_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp ogt fp128 %0, %1
  %cond = select i1 %cmp, fp128 %0, fp128 %1
  ret fp128 %cond
; CHECK-LABEL: greater_sel_qp
; CHECK: xscmpuqp [[REG:[0-9]+]]
; CHECK: bgtlr [[REG]]
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define fp128 @less_sel_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp olt fp128 %0, %1
  %cond = select i1 %cmp, fp128 %0, fp128 %1
  ret fp128 %cond
; CHECK-LABEL: less_sel_qp
; CHECK: xscmpuqp [[REG:[0-9]+]]
; CHECK: bltlr [[REG]]
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define fp128 @greater_eq_sel_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp oge fp128 %0, %1
  %cond = select i1 %cmp, fp128 %0, fp128 %1
  ret fp128 %cond
; CHECK-LABEL: greater_eq_sel_qp
; CHECK: xscmpuqp
; CHECK: crnor [[REG:[0-9]+]], {{[0-9]+}}, 0
; CHECK: bclr {{[0-9]+}}, [[REG]]
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define fp128 @less_eq_sel_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp ole fp128 %0, %1
  %cond = select i1 %cmp, fp128 %0, fp128 %1
  ret fp128 %cond
; CHECK-LABEL: less_eq_sel_qp
; CHECK: xscmpuqp
; CHECK: crnor [[REG:[0-9]+]], {{[0-9]+}}, 1
; CHECK: bclr {{[0-9]+}}, [[REG]]
; CHECK: blr
}

; Function Attrs: noinline nounwind optnone
define fp128 @equal_sel_qp() {
entry:
  %0 = load fp128, fp128* @a_qp, align 16
  %1 = load fp128, fp128* @b_qp, align 16
  %cmp = fcmp oeq fp128 %0, %1
  %cond = select i1 %cmp, fp128 %0, fp128 %1
  ret fp128 %cond
; CHECK-LABEL: equal_sel_qp
; CHECK: xscmpuqp [[REG:[0-9]+]]
; CHECK: beqlr [[REG]]
; CHECK: blr
}
