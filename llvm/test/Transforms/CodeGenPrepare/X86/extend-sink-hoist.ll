; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -cgp-branch-opts=0 -S < %s | FileCheck %s
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; The first cast should be sunk into block2, in order that the
; instruction selector can form an efficient
; ptr i64 -> i128 multiplication.
define i128 @sink(ptr %mem1, ptr %mem2) {
; CHECK-LABEL: block1:
; CHECK-NEXT: load
block1:
  %l1 = load i64, ptr %mem1
  %s1 = sext i64 %l1 to i128
  br label %block2

; CHECK-NEXT: sext
; CHECK-NEXT: load
; CHECK-NEXT: sext
block2:
  %l2 = load i64, ptr %mem2
  %s2 = sext i64 %l2 to i128
  %res = mul i128 %s1, %s2
  ret i128 %res
}

; The first cast should be hoisted into block1, in order that the
; instruction selector can form an extend-load.
define i64 @hoist(ptr %mem1, ptr %mem2) {
; CHECK-LABEL: block1:
; CHECK-NEXT: load
; CHECK-NEXT: sext
block1:
  %l1 = load i32, ptr %mem1
  br label %block2

; CHECK-NEXT: load
; CHECK-NEXT: sext
block2:
  %s1 = sext i32 %l1 to i64
  %l2 = load i32, ptr %mem2
  %s2 = sext i32 %l2 to i64
  %res = mul i64 %s1, %s2
  ret i64 %res
}

; Make sure the cast sink logic and OptimizeExtUses don't end up in an infinite
; loop.
define i128 @use_ext_source(i1 %arg) {
block1:
  %v1 = or i64 undef, undef
  %v2 = zext i64 %v1 to i128
  br i1 %arg, label %block2, label %block3

block2:
  %v3 = add i64 %v1, 1
  %v4 = zext i64 %v3 to i128
  br label %block3

block3:
  %res = phi i128 [ %v2, %block1 ], [ %v4, %block2 ]
  ret i128 %res
}

; The zext should be hoisted into block1 so the instruction selector can fold it
; with the bitcast from an i1 vector (e.g. pmovmskb/kmov).
define i32 @hoist_zext_bitcast_v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK-LABEL: @hoist_zext_bitcast_v16i8(
; CHECK-NEXT:  block1:
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq <16 x i8> %a, %b
; CHECK-NEXT:    [[BITS:%.*]] = bitcast <16 x i1> [[CMP]] to i16
; CHECK-NEXT:    [[MASK:%.*]] = zext i16 [[BITS]] to i32
; CHECK-NEXT:    [[EMPTY:%.*]] = icmp eq i16 [[BITS]], 0
; CHECK-NEXT:    br i1 [[EMPTY]], label %block3, label %block2
; CHECK:       block2:
; CHECK-NEXT:    [[TZ:%.*]] = tail call i32 @llvm.cttz.i32(i32 [[MASK]], i1 true)
; CHECK-NEXT:    ret i32 [[TZ]]
block1:
  %cmp = icmp eq <16 x i8> %a, %b
  %bits = bitcast <16 x i1> %cmp to i16
  %empty = icmp eq i16 %bits, 0
  br i1 %empty, label %block3, label %block2

block2:
  %mask = zext i16 %bits to i32
  %tz = tail call i32 @llvm.cttz.i32(i32 %mask, i1 true)
  ret i32 %tz

block3:
  ret i32 -1
}
