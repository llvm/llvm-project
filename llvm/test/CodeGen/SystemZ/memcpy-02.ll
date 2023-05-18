; Test load/store pairs that act as memcpys.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@g1src = dso_local global i8 1
@g1dst = dso_local global i8 1
@g2src = dso_local global i16 2
@g2dst = dso_local global i16 2
@g3 = dso_local global i32 3
@g4 = dso_local global i64 4
@g5src = external dso_local global fp128, align 16
@g5dst = external dso_local global fp128, align 16

; Test the simple i8 case.
define dso_local void @f1(ptr %ptr1) {
; CHECK-LABEL: f1:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, ptr %ptr1, i64 1
  %val = load i8, ptr %ptr1
  store i8 %val, ptr %ptr2
  ret void
}

; Test i8 cases where the value is zero-extended to 32 bits.
define dso_local void @f2(ptr %ptr1) {
; CHECK-LABEL: f2:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, ptr %ptr1, i64 1
  %val = load i8, ptr %ptr1
  %ext = zext i8 %val to i32
  %trunc = trunc i32 %ext to i8
  store i8 %trunc, ptr %ptr2
  ret void
}

; Test i8 cases where the value is zero-extended to 64 bits.
define dso_local void @f3(ptr %ptr1) {
; CHECK-LABEL: f3:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, ptr %ptr1, i64 1
  %val = load i8, ptr %ptr1
  %ext = zext i8 %val to i64
  %trunc = trunc i64 %ext to i8
  store i8 %trunc, ptr %ptr2
  ret void
}

; Test i8 cases where the value is sign-extended to 32 bits.
define dso_local void @f4(ptr %ptr1) {
; CHECK-LABEL: f4:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, ptr %ptr1, i64 1
  %val = load i8, ptr %ptr1
  %ext = sext i8 %val to i32
  %trunc = trunc i32 %ext to i8
  store i8 %trunc, ptr %ptr2
  ret void
}

; Test i8 cases where the value is sign-extended to 64 bits.
define dso_local void @f5(ptr %ptr1) {
; CHECK-LABEL: f5:
; CHECK: mvc 1(1,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i8, ptr %ptr1, i64 1
  %val = load i8, ptr %ptr1
  %ext = sext i8 %val to i64
  %trunc = trunc i64 %ext to i8
  store i8 %trunc, ptr %ptr2
  ret void
}

; Test the simple i16 case.
define dso_local void @f6(ptr %ptr1) {
; CHECK-LABEL: f6:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, ptr %ptr1, i64 1
  %val = load i16, ptr %ptr1
  store i16 %val, ptr %ptr2
  ret void
}

; Test i16 cases where the value is zero-extended to 32 bits.
define dso_local void @f7(ptr %ptr1) {
; CHECK-LABEL: f7:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, ptr %ptr1, i64 1
  %val = load i16, ptr %ptr1
  %ext = zext i16 %val to i32
  %trunc = trunc i32 %ext to i16
  store i16 %trunc, ptr %ptr2
  ret void
}

; Test i16 cases where the value is zero-extended to 64 bits.
define dso_local void @f8(ptr %ptr1) {
; CHECK-LABEL: f8:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, ptr %ptr1, i64 1
  %val = load i16, ptr %ptr1
  %ext = zext i16 %val to i64
  %trunc = trunc i64 %ext to i16
  store i16 %trunc, ptr %ptr2
  ret void
}

; Test i16 cases where the value is sign-extended to 32 bits.
define dso_local void @f9(ptr %ptr1) {
; CHECK-LABEL: f9:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, ptr %ptr1, i64 1
  %val = load i16, ptr %ptr1
  %ext = sext i16 %val to i32
  %trunc = trunc i32 %ext to i16
  store i16 %trunc, ptr %ptr2
  ret void
}

; Test i16 cases where the value is sign-extended to 64 bits.
define dso_local void @f10(ptr %ptr1) {
; CHECK-LABEL: f10:
; CHECK: mvc 2(2,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i16, ptr %ptr1, i64 1
  %val = load i16, ptr %ptr1
  %ext = sext i16 %val to i64
  %trunc = trunc i64 %ext to i16
  store i16 %trunc, ptr %ptr2
  ret void
}

; Test the simple i32 case.
define dso_local void @f11(ptr %ptr1) {
; CHECK-LABEL: f11:
; CHECK: mvc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32, ptr %ptr1, i64 1
  %val = load i32, ptr %ptr1
  store i32 %val, ptr %ptr2
  ret void
}

; Test i32 cases where the value is zero-extended to 64 bits.
define dso_local void @f12(ptr %ptr1) {
; CHECK-LABEL: f12:
; CHECK: mvc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32, ptr %ptr1, i64 1
  %val = load i32, ptr %ptr1
  %ext = zext i32 %val to i64
  %trunc = trunc i64 %ext to i32
  store i32 %trunc, ptr %ptr2
  ret void
}

; Test i32 cases where the value is sign-extended to 64 bits.
define dso_local void @f13(ptr %ptr1) {
; CHECK-LABEL: f13:
; CHECK: mvc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i32, ptr %ptr1, i64 1
  %val = load i32, ptr %ptr1
  %ext = sext i32 %val to i64
  %trunc = trunc i64 %ext to i32
  store i32 %trunc, ptr %ptr2
  ret void
}

; Test the i64 case.
define dso_local void @f14(ptr %ptr1) {
; CHECK-LABEL: f14:
; CHECK: mvc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i64, ptr %ptr1, i64 1
  %val = load i64, ptr %ptr1
  store i64 %val, ptr %ptr2
  ret void
}

; Test the f32 case.
define dso_local void @f15(ptr %ptr1) {
; CHECK-LABEL: f15:
; CHECK: mvc 4(4,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr float, ptr %ptr1, i64 1
  %val = load float, ptr %ptr1
  store float %val, ptr %ptr2
  ret void
}

; Test the f64 case.
define dso_local void @f16(ptr %ptr1) {
; CHECK-LABEL: f16:
; CHECK: mvc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr double, ptr %ptr1, i64 1
  %val = load double, ptr %ptr1
  store double %val, ptr %ptr2
  ret void
}

; Test the f128 case.
define dso_local void @f17(ptr %ptr1) {
; CHECK-LABEL: f17:
; CHECK: mvc 16(16,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr fp128, ptr %ptr1, i64 1
  %val = load fp128, ptr %ptr1
  store fp128 %val, ptr %ptr2
  ret void
}

; Make sure that we don't use MVC if the load is volatile.
define dso_local void @f18(ptr %ptr1) {
; CHECK-LABEL: f18:
; CHECK-NOT: mvc
; CHECK: br %r14
  %ptr2 = getelementptr i64, ptr %ptr1, i64 1
  %val = load volatile i64, ptr %ptr1
  store i64 %val, ptr %ptr2
  ret void
}

; ...likewise the store.
define dso_local void @f19(ptr %ptr1) {
; CHECK-LABEL: f19:
; CHECK-NOT: mvc
; CHECK: br %r14
  %ptr2 = getelementptr i64, ptr %ptr1, i64 1
  %val = load i64, ptr %ptr1
  store volatile i64 %val, ptr %ptr2
  ret void
}

; Test that MVC is not used for aligned loads and stores if there is
; no way of telling whether they alias.  We don't want to use MVC in
; cases where the addresses could be equal.
define dso_local void @f20(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f20:
; CHECK-NOT: mvc
; CHECK: br %r14
  %val = load i64, ptr %ptr1
  store i64 %val, ptr %ptr2
  ret void
}

; ...and again for unaligned loads and stores.
define dso_local void @f21(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f21:
; CHECK-NOT: mvc
; CHECK: br %r14
  %val = load i64, ptr %ptr1, align 2
  store i64 %val, ptr %ptr2, align 2
  ret void
}

; Test a case where there is definite overlap.
define dso_local void @f22(i64 %base) {
; CHECK-LABEL: f22:
; CHECK-NOT: mvc
; CHECK: br %r14
  %add = add i64 %base, 1
  %ptr1 = inttoptr i64 %base to ptr
  %ptr2 = inttoptr i64 %add to ptr
  %val = load i64, ptr %ptr1, align 1
  store i64 %val, ptr %ptr2, align 1
  ret void
}

; Test that we can use MVC for global addresses for i8.
define dso_local void @f23(ptr %ptr) {
; CHECK-LABEL: f23:
; CHECK-DAG: larl [[SRC:%r[0-5]]], g1src
; CHECK-DAG: larl [[DST:%r[0-5]]], g1dst
; CHECK: mvc 0(1,[[DST]]), 0([[SRC]])
; CHECK: br %r14
  %val = load i8, ptr@g1src
  store i8 %val, ptr@g1dst
  ret void
}

; Test that we use LHRL and STHRL for i16.
define dso_local void @f24(ptr %ptr) {
; CHECK-LABEL: f24:
; CHECK: lhrl [[REG:%r[0-5]]], g2src
; CHECK: sthrl [[REG]], g2dst
; CHECK: br %r14
  %val = load i16, ptr@g2src
  store i16 %val, ptr@g2dst
  ret void
}

; Test that we use LRL for i32.
define dso_local void @f25(ptr %ptr) {
; CHECK-LABEL: f25:
; CHECK: lrl [[REG:%r[0-5]]], g3
; CHECK: st [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i32, ptr@g3
  store i32 %val, ptr %ptr
  ret void
}

; ...likewise STRL.
define dso_local void @f26(ptr %ptr) {
; CHECK-LABEL: f26:
; CHECK: l [[REG:%r[0-5]]], 0(%r2)
; CHECK: strl [[REG]], g3
; CHECK: br %r14
  %val = load i32, ptr %ptr
  store i32 %val, ptr@g3
  ret void
}

; Test that we use LGRL for i64.
define dso_local void @f27(ptr %ptr) {
; CHECK-LABEL: f27:
; CHECK: lgrl [[REG:%r[0-5]]], g4
; CHECK: stg [[REG]], 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr@g4
  store i64 %val, ptr %ptr
  ret void
}

; ...likewise STGRL.
define dso_local void @f28(ptr %ptr) {
; CHECK-LABEL: f28:
; CHECK: lg [[REG:%r[0-5]]], 0(%r2)
; CHECK: stgrl [[REG]], g4
; CHECK: br %r14
  %val = load i64, ptr %ptr
  store i64 %val, ptr@g4
  ret void
}

; Test that we can use MVC for global addresses for fp128.
define dso_local void @f29(ptr %ptr) {
; CHECK-LABEL: f29:
; CHECK-DAG: larl [[SRC:%r[0-5]]], g5src
; CHECK-DAG: larl [[DST:%r[0-5]]], g5dst
; CHECK: mvc 0(16,[[DST]]), 0([[SRC]])
; CHECK: br %r14
  %val = load fp128, ptr@g5src, align 16
  store fp128 %val, ptr@g5dst, align 16
  ret void
}

; Test a case where offset disambiguation is enough.
define dso_local void @f30(ptr %ptr1) {
; CHECK-LABEL: f30:
; CHECK: mvc 8(8,%r2), 0(%r2)
; CHECK: br %r14
  %ptr2 = getelementptr i64, ptr %ptr1, i64 1
  %val = load i64, ptr %ptr1, align 1
  store i64 %val, ptr %ptr2, align 1
  ret void
}

; Test f21 in cases where TBAA tells us there is no alias.
define dso_local void @f31(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f31:
; CHECK: mvc 0(8,%r3), 0(%r2)
; CHECK: br %r14
  %val = load i64, ptr %ptr1, align 2, !tbaa !1
  store i64 %val, ptr %ptr2, align 2, !tbaa !2
  ret void
}

; Test f21 in cases where TBAA is present but doesn't help.
define dso_local void @f32(ptr %ptr1, ptr %ptr2) {
; CHECK-LABEL: f32:
; CHECK-NOT: mvc
; CHECK: br %r14
  %val = load i64, ptr %ptr1, align 2, !tbaa !1
  store i64 %val, ptr %ptr2, align 2, !tbaa !1
  ret void
}

!0 = !{ !"root" }
!1 = !{ !3, !3, i64 0 }
!2 = !{ !4, !4, i64 0 }
!3 = !{ !"set1", !0 }
!4 = !{ !"set2", !0 }
