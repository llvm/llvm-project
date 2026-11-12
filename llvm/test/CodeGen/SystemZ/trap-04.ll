; Test load-and-trap instructions (LLGFAT/LLGFTAT)
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | FileCheck %s

declare void @llvm.trap()

; Check LLGFAT with no displacement.
define i64 @f1(ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK: llgfat %r2, 0(%r2)
; CHECK: br %r14
entry:
  %val = load i32, ptr %ptr
  %ext = zext i32 %val to i64
  %cmp = icmp eq i64 %ext, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %ext
}

; Check the high end of the LLGFAT range.
define i64 @f2(ptr %src) {
; CHECK-LABEL: f2:
; CHECK: llgfat %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %val = load i32, ptr %ptr
  %ext = zext i32 %val to i64
  %cmp = icmp eq i64 %ext, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %ext
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f3(ptr %src) {
; CHECK-LABEL: f3:
; CHECK: agfi %r2, 524288
; CHECK: llgfat %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %val = load i32, ptr %ptr
  %ext = zext i32 %val to i64
  %cmp = icmp eq i64 %ext, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %ext
}

; Check that LLGFAT allows an index.
define i64 @f4(i64 %src, i64 %index) {
; CHECK-LABEL: f4:
; CHECK: llgfat %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %ext = zext i32 %val to i64
  %cmp = icmp eq i64 %ext, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %ext
}

; Check LLGTAT with no displacement.
define i64 @f5(ptr %ptr) {
; CHECK-LABEL: f5:
; CHECK: llgtat %r2, 0(%r2)
; CHECK: br %r14
entry:
  %val = load i32, ptr %ptr
  %ext = zext i32 %val to i64
  %and = and i64 %ext, 2147483647
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %and
}

; Check the high end of the LLGTAT range.
define i64 @f6(ptr %src) {
; CHECK-LABEL: f6:
; CHECK: llgtat %r2, 524284(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131071
  %val = load i32, ptr %ptr
  %ext = zext i32 %val to i64
  %and = and i64 %ext, 2147483647
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %and
}

; Check the next word up, which needs separate address logic.
; Other sequences besides this one would be OK.
define i64 @f7(ptr %src) {
; CHECK-LABEL: f7:
; CHECK: agfi %r2, 524288
; CHECK: llgtat %r2, 0(%r2)
; CHECK: br %r14
  %ptr = getelementptr i32, ptr %src, i64 131072
  %val = load i32, ptr %ptr
  %ext = zext i32 %val to i64
  %and = and i64 %ext, 2147483647
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %and
}

; Check that LLGTAT allows an index.
define i64 @f8(i64 %src, i64 %index) {
; CHECK-LABEL: f8:
; CHECK: llgtat %r2, 524287(%r3,%r2)
; CHECK: br %r14
  %add1 = add i64 %src, %index
  %add2 = add i64 %add1, 524287
  %ptr = inttoptr i64 %add2 to ptr
  %val = load i32, ptr %ptr
  %ext = zext i32 %val to i64
  %and = and i64 %ext, 2147483647
  %cmp = icmp eq i64 %and, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 %and
}

