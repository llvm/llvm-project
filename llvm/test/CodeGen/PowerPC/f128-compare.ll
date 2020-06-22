; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-unknown \
; RUN:   -enable-ppc-quad-precision -verify-machineinstrs \
; RUN:   -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr < %s | FileCheck %s

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
; CHECK: iselgt r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
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
; CHECK: isellt r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
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
; CHECK: cror 4*cr[[REG:[0-9]+]]+lt, un, lt
; CHECK: isel r{{[0-9]+}}, 0, r{{[0-9]+}}, 4*cr[[REG]]+lt
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
; CHECK: cror 4*cr[[REG:[0-9]+]]+lt, un, gt
; CHECK: isel r{{[0-9]+}}, 0, r{{[0-9]+}}, 4*cr[[REG]]+lt
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
; CHECK: iseleq r{{[0-9]+}}, r{{[0-9]+}}, r{{[0-9]+}}
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
; CHECK: iselgt r{{[0-9]+}}, 0, r{{[0-9]+}}
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
; CHECK: isellt r{{[0-9]+}}, 0, r{{[0-9]+}}
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
; CHECK: crnor 4*cr[[REG:[0-9]+]]+lt, lt, un
; CHECK: isel r{{[0-9]+}}, 0, r{{[0-9]+}}, 4*cr[[REG]]+lt
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
; CHECK: crnor 4*cr[[REG:[0-9]+]]+lt, gt, un
; CHECK: isel r{{[0-9]+}}, 0, r{{[0-9]+}}, 4*cr[[REG]]+lt
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
; CHECK: iseleq r{{[0-9]+}}, 0, r{{[0-9]+}}
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
; CHECK: xscmpuqp cr[[REG:[0-9]+]]
; CHECK: bgtlr cr[[REG]]
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
; CHECK: xscmpuqp cr[[REG:[0-9]+]]
; CHECK: bltlr cr[[REG]]
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
; CHECK: crnor 4*cr[[REG:[0-9]+]]+lt, un, lt
; CHECK: bclr {{[0-9]+}}, 4*cr[[REG]]+lt, 0
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
; CHECK: crnor 4*cr[[REG:[0-9]+]]+lt, un, gt
; CHECK: bclr {{[0-9]+}}, 4*cr[[REG]]+lt, 0
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
; CHECK: xscmpuqp cr[[REG:[0-9]+]]
; CHECK: beqlr cr[[REG]]
; CHECK: blr
}
