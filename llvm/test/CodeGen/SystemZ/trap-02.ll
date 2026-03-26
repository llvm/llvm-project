; Test zE12 conditional traps
;
; RUN: llc < %s -mtriple=s390x-linux-gnu -mcpu=zEC12 | FileCheck %s

declare void @llvm.trap()

; Check conditional compare logical and trap
define i32 @f1(i32 zeroext %a, ptr %ptr) {
; CHECK-LABEL: f1:
; CHECK:       .cfi_startproc
; CHECK-NEXT:    # %bb.0:                                # %entry
; CHECK-NEXT:    clth	%r2, 0(%r3)
; CHECK-NEXT:    lhi	%r2, 0
; CHECK-NEXT:    br	%r14
entry:
  %b = load i32, ptr %ptr
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Check conditional compare logical grande and trap
define i64 @f2(i64 zeroext %a, ptr %ptr) {
; CHECK-LABEL: f2:
; CHECK:       .cfi_startproc
; CHECK-NEXT:    # %bb.0:                                # %entry
; CHECK-NEXT:    clgtl	%r2, 0(%r3)
; CHECK-NEXT:    lghi	%r2, 0
; CHECK-NEXT:    br	%r14
entry:
  %b = load i64, ptr %ptr
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
define i32 @f3(i32 zeroext %a, ptr %base, i64 %offset) {
; CHECK-LABEL: f3:
; CHECK:       .cfi_startproc
; CHECK-NEXT:    # %bb.0:                                # %entry
; CHECK-NEXT:    sllg	%r1, %r4, 2
; CHECK-NEXT:    cl	%r2, 0(%r1,%r3)
; CHECK-NEXT:    .Ltmp0:
; CHECK-NEXT:    jh	.Ltmp0+2
; CHECK-NEXT:    lhi	%r2, 0
; CHECK-NEXT:    br	%r14
entry:
  %ptr = getelementptr i32, ptr %base, i64 %offset
  %b = load i32, ptr %ptr
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
define i64 @f4(i64 %a, ptr %base, i64 %offset) {
; CHECK-LABEL: f4:
; CHECK:       .cfi_startproc
; CHECK-NEXT:    # %bb.0:                                # %entry
; CHECK-NEXT:    sllg	%r1, %r4, 3
; CHECK-NEXT:    clg	%r2, 0(%r1,%r3)
; CHECK-NEXT:    .Ltmp1:
; CHECK-NEXT:    jh	.Ltmp1+2
; CHECK-NEXT:    lghi	%r2, 0
; CHECK-NEXT:    br	%r14
entry:
  %ptr = getelementptr i64, ptr %base, i64 %offset
  %b = load i64, ptr %ptr
  %cmp = icmp ugt i64 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 0
}

