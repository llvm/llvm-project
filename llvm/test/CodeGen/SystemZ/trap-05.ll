; Test load-and-trap instructions (LFHAT)
; See comments in asm-18.ll about testing high-word operations.
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 \
; RUN:   -no-integrated-as | FileCheck %s

declare void @llvm.trap()

; Check LAT with no displacement.
define void @f1(i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: lfhat [[REG:%r[0-9]+]], 0(%r2)
; CHECK: stepa [[REG]]
; CHECK: br %r14
entry:
  %val = load i32, i32 *%ptr
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  call void asm sideeffect "stepa $0", "h"(i32 %val)
  ret void;
}

; Check the high end of the LAT range.
define void @f2(i32 *%src) {
; CHECK-LABEL: f2:
; CHECK: lfhat [[REG:%r[0-9]+]], 524284(%r2)
; CHECK: stepa [[REG]]
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %val = load i32, i32 *%ptr
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  call void asm sideeffect "stepa $0", "h"(i32 %val)
  ret void;
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define void @f3(i32 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: lfhat [[REG:%r[0-9]+]], 0(%r2)
; CHECK: stepa [[REG]]
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %val = load i32, i32 *%ptr
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  call void asm sideeffect "stepa $0", "h"(i32 %val)
  ret void;
}

; Check that LAT allows an index.
define void @f4(i64 %src, i64 %index) {
; CHECK-LABEL: f4:
; CHECK: lfhat [[REG:%r[0-9]+]], 524287(%r3,%r2)
; CHECK: stepa [[REG]]
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i32 *
  %val = load i32, i32 *%ptr
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  call void asm sideeffect "stepa $0", "h"(i32 %val)
  ret void;
}

