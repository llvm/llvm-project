; RUN: opt -passes='require<profile-summary>,function(codegenprepare)' -S < %s | FileCheck %s
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

;; Test: basic cross-block zext hoisting.
;; This is the F14 hashmap find() pattern. The zext is hoisted to the entry
;; block so SelectionDAG can fold it with the and.
define i32 @basic_hoist(i16 %x) {
; CHECK-LABEL: @basic_hoist(
; CHECK:       entry:
; CHECK-NEXT:    [[AND:%.*]] = and i16 %x, 16383
; CHECK-NEXT:    [[WIDE:%.*]] = zext i16 [[AND]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[AND]], 0
; CHECK-NEXT:    br i1 [[CMP]]
; CHECK:       forward:
; CHECK-NEXT:    ret i32 [[WIDE]]
;
entry:
  %and = and i16 %x, 16383
  %cmp = icmp eq i16 %and, 0
  br i1 %cmp, label %else, label %forward

forward:
  %wide = zext i16 %and to i32
  br label %use

use:
  ret i32 %wide

else:
  ret i32 0
}

;; Negative: 'or' does not clear upper bits — no proof the hoist helps.
define i32 @no_hoist_or_no_known_zeros(i16 %x) {
; CHECK-LABEL: @no_hoist_or_no_known_zeros(
; CHECK:       forward:
; CHECK-NEXT:    [[WIDE:%.*]] = zext i16 [[OR:%.*]] to i32
; CHECK-NEXT:    br label %use
;
entry:
  %or = or i16 %x, 255
  br label %forward

forward:
  %wide = zext i16 %or to i32
  br label %use

use:
  ret i32 %wide
}

;; Test: i8 to i32 hoisting.
define i32 @hoist_i8_to_i32(i8 %x) {
; CHECK-LABEL: @hoist_i8_to_i32(
; CHECK:       entry:
; CHECK-NEXT:    [[AND:%.*]] = and i8 %x, 63
; CHECK-NEXT:    [[WIDE:%.*]] = zext i8 [[AND]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i8 [[AND]], 0
; CHECK-NEXT:    br i1 [[CMP]]
;
entry:
  %and = and i8 %x, 63
  %cmp = icmp eq i8 %and, 0
  br i1 %cmp, label %else, label %forward

forward:
  %wide = zext i8 %and to i32
  br label %use

use:
  ret i32 %wide

else:
  ret i32 0
}


;; Test: i16 to i64 hoisting (isZExtFree(i16, i64) is false on x86-64).
define i64 @hoist_i16_to_i64(i16 %x) {
; CHECK-LABEL: @hoist_i16_to_i64(
; CHECK:       entry:
; CHECK-NEXT:    [[AND:%.*]] = and i16 %x, 16383
; CHECK-NEXT:    [[WIDE:%.*]] = zext i16 [[AND]] to i64
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[AND]], 0
; CHECK-NEXT:    br i1 [[CMP]]
;
entry:
  %and = and i16 %x, 16383
  %cmp = icmp eq i16 %and, 0
  br i1 %cmp, label %else, label %then

then:
  %wide = zext i16 %and to i64
  ret i64 %wide

else:
  ret i64 0
}

;; Test: multiple zexts down both paths of a branch. Both should be hoisted
;; to the source block. SelectionDAG will CSE them in the same DAG.
define i32 @hoist_multi_zext_both_paths(i16 %x) {
; CHECK-LABEL: @hoist_multi_zext_both_paths(
; CHECK:       entry:
; CHECK-NEXT:    [[AND:%.*]] = and i16 %x, 16383
; CHECK-NEXT:    [[WIDE2:%.*]] = zext i16 [[AND]] to i32
; CHECK-NEXT:    [[WIDE1:%.*]] = zext i16 [[AND]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[AND]], 0
; CHECK-NEXT:    br i1 [[CMP]]
; CHECK:       then:
; CHECK-NEXT:    br label %merge
; CHECK:       else:
; CHECK-NEXT:    br label %merge
; CHECK:       merge:
; CHECK-NEXT:    [[PHI:%.*]] = phi i32 [ [[WIDE1]], %then ], [ [[WIDE2]], %else ]
; CHECK-NEXT:    ret i32 [[PHI]]
;
entry:
  %and = and i16 %x, 16383
  %cmp = icmp eq i16 %and, 0
  br i1 %cmp, label %then, label %else

then:
  %wide1 = zext i16 %and to i32
  br label %merge

else:
  %wide2 = zext i16 %and to i32
  br label %merge

merge:
  %phi = phi i32 [%wide1, %then], [%wide2, %else]
  ret i32 %phi
}


;; Test: zext block has other instructions (call) — zext still hoisted.
declare void @use(i32)
define i32 @hoist_with_call_in_block(i16 %x) {
; CHECK-LABEL: @hoist_with_call_in_block(
; CHECK:       entry:
; CHECK-NEXT:    [[AND:%.*]] = and i16 %x, 255
; CHECK-NEXT:    [[WIDE:%.*]] = zext i16 [[AND]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[AND]], 0
; CHECK-NEXT:    br i1 [[CMP]]
; CHECK:       then:
; CHECK-NEXT:    call void @use(i32 [[WIDE]])
; CHECK-NEXT:    ret i32 [[WIDE]]
;
entry:
  %and = and i16 %x, 255
  %cmp = icmp eq i16 %and, 0
  br i1 %cmp, label %else, label %then

then:
  %wide = zext i16 %and to i32
  call void @use(i32 %wide)
  ret i32 %wide

else:
  ret i32 0
}


;; ===========================================================================
;; Negative tests: cases where hoisting should NOT occur.
;; ===========================================================================

;; Negative: same block — no hoisting needed.
define i32 @same_block_no_hoist(i16 %x) {
; CHECK-LABEL: @same_block_no_hoist(
; CHECK:         [[AND:%.*]] = and i16 %x, 255
; CHECK-NEXT:    [[ZEXT:%.*]] = zext i16 [[AND]] to i32
; CHECK-NEXT:    ret i32 [[ZEXT]]
;
entry:
  %and = and i16 %x, 255
  %wide = zext i16 %and to i32
  ret i32 %wide
}

;; Test: zext block has other instructions (not a pure forwarding block).
;; The zext should still be hoisted to the source block.
define i32 @hoist_non_forwarding(i16 %x) {
; CHECK-LABEL: @hoist_non_forwarding(
; CHECK:       entry:
; CHECK-NEXT:    [[AND:%.*]] = and i16 %x, 255
; CHECK-NEXT:    [[WIDE:%.*]] = zext i16 [[AND]] to i32
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[AND]], 0
; CHECK-NEXT:    br i1 [[CMP]]
; CHECK:       then:
; CHECK-NEXT:    ret i32 [[WIDE]]
;
entry:
  %and = and i16 %x, 255
  %cmp = icmp eq i16 %and, 0
  br i1 %cmp, label %else, label %then

then:
  %wide = zext i16 %and to i32
  ret i32 %wide

else:
  ret i32 0
}

;; Negative: i32 to i64 — isZExtFree returns true on x86-64.
define i64 @no_hoist_i32_to_i64(i32 %x) {
; CHECK-LABEL: @no_hoist_i32_to_i64(
; CHECK:       forward:
; CHECK-NEXT:    [[WIDE:%.*]] = zext i32 [[AND:%.*]] to i64
; CHECK-NEXT:    br label %use
;
entry:
  %and = and i32 %x, 255
  %cmp = icmp eq i32 %and, 0
  br i1 %cmp, label %else, label %forward

forward:
  %wide = zext i32 %and to i64
  br label %use

use:
  ret i64 %wide

else:
  ret i64 0
}

;; Negative: the source also feeds arithmetic in its own block. Hoisting would
;; keep the narrow value live for the shl alongside the widened one, computing
;; the same number in two registers.
define i32 @no_hoist_narrow_use_in_source_block(i16 %x, ptr %p) {
; CHECK-LABEL: @no_hoist_narrow_use_in_source_block(
; CHECK:       entry:
; CHECK-NEXT:    [[AND:%.*]] = and i16 %x, 16383
; CHECK-NEXT:    [[SHL:%.*]] = shl i16 [[AND]], 2
; CHECK-NEXT:    store i16 [[SHL]], ptr %p
; CHECK-NEXT:    [[CMP:%.*]] = icmp eq i16 [[AND]], 0
; CHECK-NEXT:    br i1 [[CMP]]
; CHECK:       forward:
; CHECK-NEXT:    [[WIDE:%.*]] = zext i16 [[AND]] to i32
; CHECK-NEXT:    ret i32 [[WIDE]]
;
entry:
  %and = and i16 %x, 16383
  %shl = shl i16 %and, 2
  store i16 %shl, ptr %p
  %cmp = icmp eq i16 %and, 0
  br i1 %cmp, label %else, label %forward

forward:
  %wide = zext i16 %and to i32
  ret i32 %wide

else:
  ret i32 0
}
