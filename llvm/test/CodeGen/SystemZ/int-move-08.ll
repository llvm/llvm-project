; Test 32-bit GPR accesses to a PC-relative location.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@gsrc16 = dso_local global i16 1
@gsrc32 = dso_local global i32 1
@gdst16 = dso_local global i16 2
@gdst32 = dso_local global i32 2
@gsrc16u = dso_local global i16 1, align 1, section "foo"
@gsrc32u = dso_local global i32 1, align 2, section "foo"
@gdst16u = dso_local global i16 2, align 1, section "foo"
@gdst32u = dso_local global i32 2, align 2, section "foo"
@garray8 = dso_local global [2 x i8] [i8 100, i8 101]
@garray16 = dso_local global [2 x i16] [i16 102, i16 103]

; Check sign-extending loads from i16.
define dso_local i32 @f1() {
; CHECK-LABEL: f1:
; CHECK: lhrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16, ptr@gsrc16
  %ext = sext i16 %val to i32
  ret i32 %ext
}

; Check zero-extending loads from i16.
define dso_local i32 @f2() {
; CHECK-LABEL: f2:
; CHECK: llhrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16, ptr@gsrc16
  %ext = zext i16 %val to i32
  ret i32 %ext
}

; Check truncating 16-bit stores.
define dso_local void @f3(i32 %val) {
; CHECK-LABEL: f3:
; CHECK: sthrl %r2, gdst16
; CHECK: br %r14
  %half = trunc i32 %val to i16
  store i16 %half, ptr@gdst16
  ret void
}

; Check plain loads and stores.
define dso_local void @f4() {
; CHECK-LABEL: f4:
; CHECK: lrl %r0, gsrc32
; CHECK: strl %r0, gdst32
; CHECK: br %r14
  %val = load i32, ptr@gsrc32
  store i32 %val, ptr@gdst32
  ret void
}

; Repeat f1 with an unaligned variable.
define dso_local i32 @f5() {
; CHECK-LABEL: f5:
; CHECK: lgrl [[REG:%r[0-5]]], gsrc16u
; CHECK: lh %r2, 0([[REG]])
; CHECK: br %r14
  %val = load i16, ptr@gsrc16u, align 1
  %ext = sext i16 %val to i32
  ret i32 %ext
}

; Repeat f2 with an unaligned variable.
define dso_local i32 @f6() {
; CHECK-LABEL: f6:
; CHECK: lgrl [[REG:%r[0-5]]], gsrc16u
; CHECK: llh %r2, 0([[REG]])
; CHECK: br %r14
  %val = load i16, ptr@gsrc16u, align 1
  %ext = zext i16 %val to i32
  ret i32 %ext
}

; Repeat f3 with an unaligned variable.
define dso_local void @f7(i32 %val) {
; CHECK-LABEL: f7:
; CHECK: lgrl [[REG:%r[0-5]]], gdst16u
; CHECK: sth %r2, 0([[REG]])
; CHECK: br %r14
  %half = trunc i32 %val to i16
  store i16 %half, ptr@gdst16u, align 1
  ret void
}

; Repeat f4 with unaligned variables.
define dso_local void @f8() {
; CHECK-LABEL: f8:
; CHECK: larl [[REG:%r[0-5]]], gsrc32u
; CHECK: l [[VAL:%r[0-5]]], 0([[REG]])
; CHECK: larl [[REG:%r[0-5]]], gdst32u
; CHECK: st [[VAL]], 0([[REG]])
; CHECK: br %r14
  %val = load i32, ptr@gsrc32u, align 2
  store i32 %val, ptr@gdst32u, align 2
  ret void
}

; Test a case where we want to use one LARL for accesses to two different
; parts of a variable.
define dso_local void @f9() {
; CHECK-LABEL: f9:
; CHECK: larl [[REG:%r[0-5]]], garray8
; CHECK: llc [[VAL:%r[0-5]]], 0([[REG]])
; CHECK: srl [[VAL]], 1
; CHECK: stc [[VAL]], 1([[REG]])
; CHECK: br %r14
  %ptr1 = getelementptr [2 x i8], ptr@garray8, i64 0, i64 0
  %ptr2 = getelementptr [2 x i8], ptr@garray8, i64 0, i64 1
  %val = load i8, ptr %ptr1
  %shr = lshr i8 %val, 1
  store i8 %shr, ptr %ptr2
  ret void
}

; Test a case where we want to use separate relative-long addresses for
; two different parts of a variable.
define dso_local void @f10() {
; CHECK-LABEL: f10:
; CHECK: llhrl [[VAL:%r[0-5]]], garray16
; CHECK: srl [[VAL]], 1
; CHECK: sthrl [[VAL]], garray16+2
; CHECK: br %r14
  %ptr1 = getelementptr [2 x i16], ptr@garray16, i64 0, i64 0
  %ptr2 = getelementptr [2 x i16], ptr@garray16, i64 0, i64 1
  %val = load i16, ptr %ptr1
  %shr = lshr i16 %val, 1
  store i16 %shr, ptr %ptr2
  ret void
}
