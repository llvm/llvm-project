; Test zE12 conditional traps
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | FileCheck %s

declare void @llvm.trap()

; Check conditional compare logical and trap
define i32 @f1(i32 zeroext %a, i32 *%ptr) {
; CHECK-LABEL: f1:
; CHECK: clth %r2, 0(%r3)
; CHECK: lhi %r2, 0
; CHECK: br %r14
entry:
  %b = load i32, i32 *%ptr
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Check conditional compare logical grande and trap
define i64 @f2(i64 zeroext %a, i64 *%ptr) {
; CHECK-LABEL: f2:
; CHECK: clgtl %r2, 0(%r3)
; CHECK: lghi %r2, 0
; CHECK: br %r14
entry:
  %b = load i64, i64 *%ptr
  %cmp = icmp ult i64 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 0
}

; Verify that we don't attempt to use the compare and trap
; instruction with an index operand.
define i32 @f3(i32 zeroext %a, i32 *%base, i64 %offset) {
; CHECK-LABEL: f3:
; CHECK: cl %r2, 0(%r{{[0-5]}},%r3)
; CHECK-LABEL: .Ltmp0
; CHECK: jh .Ltmp0+2
; CHECK: lhi %r2, 0
; CHECK: br %r14
entry:
  %ptr = getelementptr i32, i32 *%base, i64 %offset
  %b = load i32, i32 *%ptr
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Verify that we don't attempt to use the compare and trap grande
; instruction with an index operand.
define i64 @f4(i64 %a, i64 *%base, i64 %offset) {
; CHECK-LABEL: f4:
; CHECK: clg %r2, 0(%r{{[0-5]}},%r3)
; CHECK-LABEL: .Ltmp1
; CHECK: jh .Ltmp1+2
; CHECK: lghi %r2, 0
; CHECK: br %r14
entry:
  %ptr = getelementptr i64, i64 *%base, i64 %offset
  %b = load i64, i64 *%ptr
  %cmp = icmp ugt i64 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 0
}

