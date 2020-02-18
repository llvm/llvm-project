; RUN: llc -mtriple=arm-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V4
; RUN: llc -mtriple=armv6-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V6
; RUN: llc -mtriple=armv7-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V6
; RUN: llc -mtriple=thumb-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-THUMB
; RUN: llc -mtriple=thumbv6-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-THUMB
; RUN: llc -mtriple=thumbv6t2-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-THUMBV6T2
; RUN: llc -mtriple=thumbv7-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-THUMBV6T2
; RUN: llc -mtriple=thumbv7m-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-V4
; RUN: llc -mtriple=thumbv7em-eabi %s -o - | FileCheck %s -check-prefix=CHECK -check-prefix=CHECK-THUMBV6T2

; Next test would previously trigger an assertion responsible for verification of
; call site info state.
; RUN: llc -stop-after=if-converter -debug-entry-values -mtriple=thumbv6t2-eabi %s -o -| FileCheck %s -check-prefix=CHECK-CALLSITE
; CHECK-CALLSITE: name:  test_used_flags
; CHECK-CALLSITE: callSites:


define i32 @Test0(i32 %a, i32 %b, i32 %c) nounwind readnone ssp {
entry:
; CHECK-LABEL: Test0
; CHECK-NOT: smmls
  %conv4 = zext i32 %a to i64
  %conv1 = sext i32 %b to i64
  %conv2 = sext i32 %c to i64
  %mul = mul nsw i64 %conv2, %conv1
  %shr5 = lshr i64 %mul, 32
  %sub = sub nsw i64 %conv4, %shr5
  %conv3 = trunc i64 %sub to i32
  ret i32 %conv3
}

define i32 @Test1(i32 %a, i32 %b, i32 %c) {
;CHECK-LABEL: Test1
;CHECK-V4-NOT: smmls
;CHECK-THUMB-NOT: smmls
;CHECK-V6: smmls r0, [[Rn:r[1-2]]], [[Rm:r[1-2]]], r0
;CHECK-THUMBV6T2: smmls r0, [[Rn:r[1-2]]], [[Rm:r[1-2]]], r0
entry:
  %conv = sext i32 %b to i64
  %conv1 = sext i32 %c to i64
  %mul = mul nsw i64 %conv1, %conv
  %conv26 = zext i32 %a to i64
  %shl = shl nuw i64 %conv26, 32
  %sub = sub nsw i64 %shl, %mul
  %shr7 = lshr i64 %sub, 32
  %conv3 = trunc i64 %shr7 to i32
  ret i32 %conv3
}

declare void @opaque(i32)
define void @test_used_flags(i32 %in1, i32 %in2) {
; CHECK-LABEL: test_used_flags:
; CHECK-THUMB: movs    r2, #0
; CHECK-THUMB: rsbs    r0, r0, #0
; CHECK-THUMB: sbcs    r2, r1
; CHECK-THUMB: bge
; CHECK-V6: smull [[PROD_LO:r[0-9]+]], [[PROD_HI:r[0-9]+]], r0, r1
; CHECK-V6: rsbs {{.*}}, [[PROD_LO]], #0
; CHECK-V6: rscs {{.*}}, [[PROD_HI]], #0
; CHECK-THUMBV6T2: smull [[PROD_LO:r[0-9]+]], [[PROD_HI:r[0-9]+]], r0, r1
; CHECK-THUMBV6T2: movs	[[ZERO:r[0-9]+]], #0
; CHECK-THUMBV6T2: rsbs	{{.*}}, [[PROD_LO]], #0
; CHECK-THUMBV6T2: sbcs.w {{.*}}, [[ZERO]], [[PROD_HI]]
  %in1.64 = sext i32 %in1 to i64
  %in2.64 = sext i32 %in2 to i64
  %mul = mul nsw i64 %in1.64, %in2.64
  %tst = icmp slt i64 %mul, 1
  br i1 %tst, label %true, label %false

true:
  call void @opaque(i32 42)
  ret void

false:
  call void @opaque(i32 56)
  ret void
}
