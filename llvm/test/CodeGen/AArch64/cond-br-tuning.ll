; RUN: llc -debugify-and-strip-all-safe < %s -O3 -mtriple=aarch64-eabi -verify-machineinstrs | FileCheck %s

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-linaro-linux-gnueabi"

; CMN is an alias of ADDS.
; CHECK-LABEL: test_add_cbz:
; CHECK: cmn w0, w1
; CHECK: b.eq
; CHECK: ret
define void @test_add_cbz(i32 %a, i32 %b, i32* %ptr) {
  %c = add nsw i32 %a, %b
  %d = icmp ne i32 %c, 0
  br i1 %d, label %L1, label %L2
L1:
  store i32 0, i32* %ptr, align 4
  ret void
L2:
  store i32 1, i32* %ptr, align 4
  ret void
}

; CHECK-LABEL: test_add_cbz_multiple_use:
; CHECK: adds
; CHECK: b.eq
; CHECK: ret
define void @test_add_cbz_multiple_use(i32 %a, i32 %b, i32* %ptr) {
  %c = add nsw i32 %a, %b
  %d = icmp ne i32 %c, 0
  br i1 %d, label %L1, label %L2
L1:
  store i32 0, i32* %ptr, align 4
  ret void
L2:
  store i32 %c, i32* %ptr, align 4
  ret void
}

; CHECK-LABEL: test_add_cbz_64:
; CHECK: cmn x0, x1
; CHECK: b.eq
define void @test_add_cbz_64(i64 %a, i64 %b, i64* %ptr) {
  %c = add nsw i64 %a, %b
  %d = icmp ne i64 %c, 0
  br i1 %d, label %L1, label %L2
L1:
  store i64 0, i64* %ptr, align 4
  ret void
L2:
  store i64 1, i64* %ptr, align 4
  ret void
}

; CHECK-LABEL: test_and_cbz:
; CHECK: tst w0, #0x6
; CHECK: b.eq
define void @test_and_cbz(i32 %a, i32* %ptr) {
  %c = and i32 %a, 6
  %d = icmp ne i32 %c, 0
  br i1 %d, label %L1, label %L2
L1:
  store i32 0, i32* %ptr, align 4
  ret void
L2:
  store i32 1, i32* %ptr, align 4
  ret void
}

; CHECK-LABEL: test_bic_cbnz:
; CHECK: bics wzr, w1, w0
; CHECK: b.ne
define void @test_bic_cbnz(i32 %a, i32 %b, i32* %ptr) {
  %c = and i32 %a, %b
  %d = icmp eq i32 %c, %b
  br i1 %d, label %L1, label %L2
L1:
  store i32 0, i32* %ptr, align 4
  ret void
L2:
  store i32 1, i32* %ptr, align 4
  ret void
}

; CHECK-LABEL: test_add_tbz:
; CHECK: adds
; CHECK: b.pl
; CHECK: ret
define void @test_add_tbz(i32 %a, i32 %b, i32* %ptr) {
entry:
  %add = add nsw i32 %a, %b
  %cmp36 = icmp sge i32 %add, 0
  br i1 %cmp36, label %L2, label %L1
L1:
  store i32 %add, i32* %ptr, align 8
  br label %L2
L2:
  ret void
}

; CHECK-LABEL: test_subs_tbz:
; CHECK: subs
; CHECK: b.pl
; CHECK: ret
define void @test_subs_tbz(i32 %a, i32 %b, i32* %ptr) {
entry:
  %sub = sub nsw i32 %a, %b
  %cmp36 = icmp sge i32 %sub, 0
  br i1 %cmp36, label %L2, label %L1
L1:
  store i32 %sub, i32* %ptr, align 8
  br label %L2
L2:
  ret void
}

; CHECK-LABEL: test_add_tbnz
; CHECK: adds
; CHECK: b.mi
; CHECK: ret
define void @test_add_tbnz(i32 %a, i32 %b, i32* %ptr) {
entry:
  %add = add nsw i32 %a, %b
  %cmp36 = icmp slt i32 %add, 0
  br i1 %cmp36, label %L2, label %L1
L1:
  store i32 %add, i32* %ptr, align 8
  br label %L2
L2:
  ret void
}

; CHECK-LABEL: test_subs_tbnz
; CHECK: subs
; CHECK: b.mi
; CHECK: ret
define void @test_subs_tbnz(i32 %a, i32 %b, i32* %ptr) {
entry:
  %sub = sub nsw i32 %a, %b
  %cmp36 = icmp slt i32 %sub, 0
  br i1 %cmp36, label %L2, label %L1
L1:
  store i32 %sub, i32* %ptr, align 8
  br label %L2
L2:
  ret void
}

declare void @foo()
declare void @bar(i32)

; Don't transform since the call will clobber the NZCV bits.
; CHECK-LABEL: test_call_clobber:
; CHECK: and w[[DST:[0-9]+]], w1, #0x6
; CHECK: bl bar
; CHECK: cbnz w[[DST]]
define void @test_call_clobber(i32 %unused, i32 %a) {
entry:
  %c = and i32 %a, 6
  call void @bar(i32 %c)
  %tobool = icmp eq i32 %c, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:
  tail call void @foo()
  unreachable

if.end:
  ret void
}
