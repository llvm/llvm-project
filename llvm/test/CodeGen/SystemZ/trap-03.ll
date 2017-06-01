; Test load-and-trap instructions (LAT/LGAT)
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | FileCheck %s

declare void @llvm.trap()

; Check LAT with no displacement.
define i32 @f1(i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: lat %r2, 0(%r2)
; CHECK: br %r14
entry:
  %val = load i32, i32 *%ptr
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 %val
}

; Check the high end of the LAT range.
define i32 @f2(i32 *%src) {
; CHECK-LABEL: f2:
; CHECK: lat %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131071
  %val = load i32, i32 *%ptr
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 %val
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i32 @f3(i32 *%src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: lat %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, i32 *%src, i64 131072
  %val = load i32, i32 *%ptr
  %cmp = icmp eq i32 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 %val
}

; Check that LAT allows an index.
define i32 @f4(i64 %src, i64 %index) {
; CHECK-LABEL: f4:
; CHECK: lat %r2, 524287(%r3,%r2)
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
  ret i32 %val
}

; Check LGAT with no displacement.
define i64 @f5(i64 *%ptr) {
; CHECK-LABEL: f5:
; CHECK: lgat %r2, 0(%r2)
; CHECK: br %r14
entry:
  %val = load i64, i64 *%ptr
  %cmp = icmp eq i64 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %val
}

; Check the high end of the LGAT range.
define i64 @f6(i64 *%src) {
; CHECK-LABEL: f6:
; CHECK: lgat %r2, 524280(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65535
  %val = load i64, i64 *%ptr
  %cmp = icmp eq i64 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %val
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(i64 *%src) {
; CHECK-LABEL: f7:
; CHECK: agfi %r2, 524288
; CHECK: lgat %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i64, i64 *%src, i64 65536
  %val = load i64, i64 *%ptr
  %cmp = icmp eq i64 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %val
}

; Check that LGAT allows an index.
define i64 @f8(i64 %src, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: lgat %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to i64 *
  %val = load i64, i64 *%ptr
  %cmp = icmp eq i64 %val, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %val
}
