; Test 64-bit GPR accesses to a PC-relative location.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

@gsrc16 = global i16 1
@gsrc32 = global i32 1
@gsrc64 = global i64 1
@gdst16 = global i16 2
@gdst32 = global i32 2
@gdst64 = global i64 2
@gsrc16u = global i16 1, align 1, section "foo"
@gsrc32u = global i32 1, align 2, section "foo"
@gsrc64u = global i64 1, align 4, section "foo"
@gdst16u = global i16 2, align 1, section "foo"
@gdst32u = global i32 2, align 2, section "foo"
@gdst64u = global i64 2, align 4, section "foo"

; Check sign-extending loads from i16.
define i64 @f1() {
; CHECK-LABEL: f1:
; CHECK: lghrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16, i16 *@gsrc16
  %ext = sext i16 %val to i64
  ret i64 %ext
}

; Check zero-extending loads from i16.
define i64 @f2() {
; CHECK-LABEL: f2:
; CHECK: llghrl %r2, gsrc16
; CHECK: br %r14
  %val = load i16, i16 *@gsrc16
  %ext = zext i16 %val to i64
  ret i64 %ext
}

; Check sign-extending loads from i32.
define i64 @f3() {
; CHECK-LABEL: f3:
; CHECK: lgfrl %r2, gsrc32
; CHECK: br %r14
  %val = load i32, i32 *@gsrc32
  %ext = sext i32 %val to i64
  ret i64 %ext
}

; Check zero-extending loads from i32.
define i64 @f4() {
; CHECK-LABEL: f4:
; CHECK: llgfrl %r2, gsrc32
; CHECK: br %r14
  %val = load i32, i32 *@gsrc32
  %ext = zext i32 %val to i64
  ret i64 %ext
}

; Check truncating 16-bit stores.
define void @f5(i64 %val) {
; CHECK-LABEL: f5:
; CHECK: sthrl %r2, gdst16
; CHECK: br %r14
  %half = trunc i64 %val to i16
  store i16 %half, i16 *@gdst16
  ret void
}

; Check truncating 32-bit stores.
define void @f6(i64 %val) {
; CHECK-LABEL: f6:
; CHECK: strl %r2, gdst32
; CHECK: br %r14
  %word = trunc i64 %val to i32
  store i32 %word, i32 *@gdst32
  ret void
}

; Check plain loads and stores.
define void @f7() {
; CHECK-LABEL: f7:
; CHECK: lgrl %r0, gsrc64
; CHECK: stgrl %r0, gdst64
; CHECK: br %r14
  %val = load i64, i64 *@gsrc64
  store i64 %val, i64 *@gdst64
  ret void
}

; Repeat f1 with an unaligned variable.
define i64 @f8() {
; CHECK-LABEL: f8:
; CHECK: lgrl [[REG:%r[0-5]]], gsrc16u@GOT
; CHECK: lgh %r2, 0([[REG]])
; CHECK: br %r14
  %val = load i16, i16 *@gsrc16u, align 1
  %ext = sext i16 %val to i64
  ret i64 %ext
}

; Repeat f2 with an unaligned variable.
define i64 @f9() {
; CHECK-LABEL: f9:
; CHECK: lgrl [[REG:%r[0-5]]], gsrc16u@GOT
; CHECK: llgh %r2, 0([[REG]])
; CHECK: br %r14
  %val = load i16, i16 *@gsrc16u, align 1
  %ext = zext i16 %val to i64
  ret i64 %ext
}

; Repeat f3 with an unaligned variable.
define i64 @f10() {
; CHECK-LABEL: f10:
; CHECK: larl [[REG:%r[0-5]]], gsrc32u
; CHECK: lgf %r2, 0([[REG]])
; CHECK: br %r14
  %val = load i32, i32 *@gsrc32u, align 2
  %ext = sext i32 %val to i64
  ret i64 %ext
}

; Repeat f4 with an unaligned variable.
define i64 @f11() {
; CHECK-LABEL: f11:
; CHECK: larl [[REG:%r[0-5]]], gsrc32u
; CHECK: llgf %r2, 0([[REG]])
; CHECK: br %r14
  %val = load i32, i32 *@gsrc32u, align 2
  %ext = zext i32 %val to i64
  ret i64 %ext
}

; Repeat f5 with an unaligned variable.
define void @f12(i64 %val) {
; CHECK-LABEL: f12:
; CHECK: lgrl [[REG:%r[0-5]]], gdst16u@GOT
; CHECK: sth %r2, 0([[REG]])
; CHECK: br %r14
  %half = trunc i64 %val to i16
  store i16 %half, i16 *@gdst16u, align 1
  ret void
}

; Repeat f6 with an unaligned variable.
define void @f13(i64 %val) {
; CHECK-LABEL: f13:
; CHECK: larl [[REG:%r[0-5]]], gdst32u
; CHECK: st %r2, 0([[REG]])
; CHECK: br %r14
  %word = trunc i64 %val to i32
  store i32 %word, i32 *@gdst32u, align 2
  ret void
}

; Repeat f7 with unaligned variables.
define void @f14() {
; CHECK-LABEL: f14:
; CHECK: larl [[REG:%r[0-5]]], gsrc64u
; CHECK: lg [[VAL:%r[0-5]]], 0([[REG]])
; CHECK: larl [[REG:%r[0-5]]], gdst64u
; CHECK: stg [[VAL]], 0([[REG]])
; CHECK: br %r14
  %val = load i64, i64 *@gsrc64u, align 4
  store i64 %val, i64 *@gdst64u, align 4
  ret void
}
