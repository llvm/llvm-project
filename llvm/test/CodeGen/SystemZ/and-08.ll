; Test memory-to-memory ANDs.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g1src = dso_local global i8 1
@g1dst = dso_local global i8 1
@g2src = dso_local global i16 2
@g2dst = dso_local global i16 2

; Test the simple i8 case.
define dso_local void @f1(i8 *%ptr1) {
; CHECK-LABEL: f1:
; CHECK: nc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, i8 *%ptr1, i64 1
  %val = load i8, i8 *%ptr1
  %old = load i8, i8 *%ptr2
  %and = and i8 %val, %old
  store i8 %and, i8 *%ptr2
  ret void
}

; ...and again in reverse.
define dso_local void @f2(i8 *%ptr1) {
; CHECK-LABEL: f2:
; CHECK: nc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, i8 *%ptr1, i64 1
  %val = load i8, i8 *%ptr1
  %old = load i8, i8 *%ptr2
  %and = and i8 %old, %val
  store i8 %and, i8 *%ptr2
  ret void
}

; Test i8 cases where one value is zero-extended to 32 bits and the other
; sign-extended.
define dso_local void @f3(i8 *%ptr1) {
; CHECK-LABEL: f3:
; CHECK: nc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, i8 *%ptr1, i64 1
  %val = load i8, i8 *%ptr1
  %extval = zext i8 %val to i32
  %old = load i8, i8 *%ptr2
  %extold = sext i8 %old to i32
  %and = and i32 %extval, %extold
  %trunc = trunc i32 %and to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; ...and again with the extension types reversed.
define dso_local void @f4(i8 *%ptr1) {
; CHECK-LABEL: f4:
; CHECK: nc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, i8 *%ptr1, i64 1
  %val = load i8, i8 *%ptr1
  %extval = sext i8 %val to i32
  %old = load i8, i8 *%ptr2
  %extold = zext i8 %old to i32
  %and = and i32 %extval, %extold
  %trunc = trunc i32 %and to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; ...and again with two sign extensions.
define dso_local void @f5(i8 *%ptr1) {
; CHECK-LABEL: f5:
; CHECK: nc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, i8 *%ptr1, i64 1
  %val = load i8, i8 *%ptr1
  %extval = sext i8 %val to i32
  %old = load i8, i8 *%ptr2
  %extold = sext i8 %old to i32
  %and = and i32 %extval, %extold
  %trunc = trunc i32 %and to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; ...and again with two zero extensions.
define dso_local void @f6(i8 *%ptr1) {
; CHECK-LABEL: f6:
; CHECK: nc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, i8 *%ptr1, i64 1
  %val = load i8, i8 *%ptr1
  %extval = zext i8 %val to i32
  %old = load i8, i8 *%ptr2
  %extold = zext i8 %old to i32
  %and = and i32 %extval, %extold
  %trunc = trunc i32 %and to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; Test i8 cases where the value is extended to 64 bits (just one case
; this time).
define dso_local void @f7(i8 *%ptr1) {
; CHECK-LABEL: f7:
; CHECK: nc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, i8 *%ptr1, i64 1
  %val = load i8, i8 *%ptr1
  %extval = sext i8 %val to i64
  %old = load i8, i8 *%ptr2
  %extold = zext i8 %old to i64
  %and = and i64 %extval, %extold
  %trunc = trunc i64 %and to i8
  store i8 %trunc, i8 *%ptr2
  ret void
}

; Test the simple i16 case.
define dso_local void @f8(i16 *%ptr1) {
; CHECK-LABEL: f8:
; CHECK: nc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, i16 *%ptr1, i64 1
  %val = load i16, i16 *%ptr1
  %old = load i16, i16 *%ptr2
  %and = and i16 %val, %old
  store i16 %and, i16 *%ptr2
  ret void
}

; Test i16 cases where the value is extended to 32 bits.
define dso_local void @f9(i16 *%ptr1) {
; CHECK-LABEL: f9:
; CHECK: nc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, i16 *%ptr1, i64 1
  %val = load i16, i16 *%ptr1
  %extval = zext i16 %val to i32
  %old = load i16, i16 *%ptr2
  %extold = sext i16 %old to i32
  %and = and i32 %extval, %extold
  %trunc = trunc i32 %and to i16
  store i16 %trunc, i16 *%ptr2
  ret void
}

; Test i16 cases where the value is extended to 64 bits.
define dso_local void @f10(i16 *%ptr1) {
; CHECK-LABEL: f10:
; CHECK: nc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, i16 *%ptr1, i64 1
  %val = load i16, i16 *%ptr1
  %extval = sext i16 %val to i64
  %old = load i16, i16 *%ptr2
  %extold = zext i16 %old to i64
  %and = and i64 %extval, %extold
  %trunc = trunc i64 %and to i16
  store i16 %trunc, i16 *%ptr2
  ret void
}

; Test the simple i32 case.
define dso_local void @f11(i32 *%ptr1) {
; CHECK-LABEL: f11:
; CHECK: nc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32, i32 *%ptr1, i64 1
  %val = load i32, i32 *%ptr1
  %old = load i32, i32 *%ptr2
  %and = and i32 %old, %val
  store i32 %and, i32 *%ptr2
  ret void
}

; Test i32 cases where the value is extended to 64 bits.
define dso_local void @f12(i32 *%ptr1) {
; CHECK-LABEL: f12:
; CHECK: nc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32, i32 *%ptr1, i64 1
  %val = load i32, i32 *%ptr1
  %extval = sext i32 %val to i64
  %old = load i32, i32 *%ptr2
  %extold = zext i32 %old to i64
  %and = and i64 %extval, %extold
  %trunc = trunc i64 %and to i32
  store i32 %trunc, i32 *%ptr2
  ret void
}

; Test the i64 case.
define dso_local void @f13(i64 *%ptr1) {
; CHECK-LABEL: f13:
; CHECK: nc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i64, i64 *%ptr1, i64 1
  %val = load i64, i64 *%ptr1
  %old = load i64, i64 *%ptr2
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2
  ret void
}

; Make sure that we don't use NC if the first load is volatile.
define dso_local void @f14(i64 *%ptr1) {
; CHECK-LABEL: f14:
; CHECK-NOT: nc
; CHECK: br %r14
  %ptr2 = getelementptr i64, i64 *%ptr1, i64 1
  %val = load volatile i64, i64 *%ptr1
  %old = load i64, i64 *%ptr2
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2
  ret void
}

; ...likewise the second.
define dso_local void @f15(i64 *%ptr1) {
; CHECK-LABEL: f15:
; CHECK-NOT: nc
; CHECK: br %r14
  %ptr2 = getelementptr i64, i64 *%ptr1, i64 1
  %val = load i64, i64 *%ptr1
  %old = load volatile i64, i64 *%ptr2
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2
  ret void
}

; ...likewise the store.
define dso_local void @f16(i64 *%ptr1) {
; CHECK-LABEL: f16:
; CHECK-NOT: nc
; CHECK: br %r14
  %ptr2 = getelementptr i64, i64 *%ptr1, i64 1
  %val = load i64, i64 *%ptr1
  %old = load i64, i64 *%ptr2
  %and = and i64 %old, %val
  store volatile i64 %and, i64 *%ptr2
  ret void
}

; Test that NC is not used for aligned loads and stores if there is
; no way of telling whether they alias.  We don't want to use NC in
; cases where the addresses could be equal.
define dso_local void @f17(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f17:
; CHECK-NOT: nc
; CHECK: br %r14
  %val = load i64, i64 *%ptr1
  %old = load i64, i64 *%ptr2
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2
  ret void
}

; ...but if one of the loads isn't aligned, we can't be sure.
define dso_local void @f18(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f18:
; CHECK-NOT: nc
; CHECK: br %r14
  %val = load i64, i64 *%ptr1, align 2
  %old = load i64, i64 *%ptr2
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2
  ret void
}

; Repeat the previous test with the operands in the opposite order.
define dso_local void @f19(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f19:
; CHECK-NOT: nc
; CHECK: br %r14
  %val = load i64, i64 *%ptr1, align 2
  %old = load i64, i64 *%ptr2
  %and = and i64 %val, %old
  store i64 %and, i64 *%ptr2
  ret void
}

; ...and again with the other operand being unaligned.
define dso_local void @f20(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f20:
; CHECK-NOT: nc
; CHECK: br %r14
  %val = load i64, i64 *%ptr1
  %old = load i64, i64 *%ptr2, align 2
  %and = and i64 %val, %old
  store i64 %and, i64 *%ptr2, align 2
  ret void
}

; Test a case where there is definite overlap.
define dso_local void @f21(i64 %base) {
; CHECK-LABEL: f21:
; CHECK-NOT: nc
; CHECK: br %r14
  %add = add i64 %base, 1
  %ptr1 = inttoptr i64 %base to i64 *
  %ptr2 = inttoptr i64 %add to i64 *
  %val = load i64, i64 *%ptr1
  %old = load i64, i64 *%ptr2, align 1
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2, align 1
  ret void
}

; Test that we can use NC for global addresses for i8.
define dso_local void @f22(i8 *%ptr) {
; CHECK-LABEL: f22:
; CHECK-DAG: larl [[SRC:%r[0-5]]], g1src
; CHECK-DAG: larl [[DST:%r[0-5]]], g1dst
; CHECK: nc 0(1,[[DST]]), 0([[SRC]])
; CHECK: br %r14
  %val = load i8, i8 *@g1src
  %old = load i8, i8 *@g1dst
  %and = and i8 %val, %old
  store i8 %and, i8 *@g1dst
  ret void
}

; Test that we use NC even where LHRL and STHRL are available.
define dso_local void @f23(i16 *%ptr) {
; CHECK-LABEL: f23:
; CHECK-DAG: larl [[SRC:%r[0-5]]], g2src
; CHECK-DAG: larl [[DST:%r[0-5]]], g2dst
; CHECK: nc 0(2,[[DST]]), 0([[SRC]])
; CHECK: br %r14
  %val = load i16, i16 *@g2src
  %old = load i16, i16 *@g2dst
  %and = and i16 %val, %old
  store i16 %and, i16 *@g2dst
  ret void
}

; Test a case where offset disambiguation is enough.
define dso_local void @f24(i64 *%ptr1) {
; CHECK-LABEL: f24:
; CHECK: nc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i64, i64 *%ptr1, i64 1
  %val = load i64, i64 *%ptr1, align 1
  %old = load i64, i64 *%ptr2, align 1
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2, align 1
  ret void
}

; Test a case where TBAA tells us there is no alias.
define dso_local void @f25(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f25:
; CHECK: nc 0(8,%r3), 0(%r2)
; CHECK: br %r14
  %val = load i64, i64 *%ptr1, align 2, !tbaa !3
  %old = load i64, i64 *%ptr2, align 2, !tbaa !4
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2, align 2, !tbaa !4
  ret void
}

; Test a case where TBAA information is present but doesn't help.
define dso_local void @f26(i64 *%ptr1, i64 *%ptr2) {
; CHECK-LABEL: f26:
; CHECK-NOT: nc
; CHECK: br %r14
  %val = load i64, i64 *%ptr1, align 2, !tbaa !3
  %old = load i64, i64 *%ptr2, align 2, !tbaa !3
  %and = and i64 %old, %val
  store i64 %and, i64 *%ptr2, align 2, !tbaa !3
  ret void
}

; Test a case where one of the loads are optimized by the DAGCombiner to a
; zero-extending load of half the original size.
define dso_local void @f27(i16* noalias %ptr1, i16* noalias %ptr2) {
; CHECK-LABEL: f27:
; CHECK-NOT: nc
; CHECK: br %r14
entry:
  %0 = load i16, i16 *%ptr1, align 2
  %1 = lshr i16 %0, 8
  %2 = load i16, i16 *%ptr2, align 2
  %and7 = and i16 %1, %2
  store i16 %and7, i16 *%ptr1, align 2
  ret void
}

!0 = !{ !"root" }
!1 = !{ !"set1", !0 }
!2 = !{ !"set2", !0 }
!3 = !{ !1, !1, i64 0}
!4 = !{ !2, !2, i64 0}
