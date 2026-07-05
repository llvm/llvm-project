; Test traps and conditional traps
;
; RUN: llc < %s -mtriple=s390x-linux-gnu | FileCheck %s

declare void @llvm.trap()

; Check unconditional traps
define i32 @f0() {
; CHECK-LABEL: f0:
; CHECK-LABEL: .Ltmp0
; CHECK: j .Ltmp0+2
entry:
  tail call void @llvm.trap()
  ret i32 0
}

; Check conditional compare immediate and trap
define i32 @f1(i32 signext %a) {
; CHECK-LABEL: f1:
; CHECK: cithe %r2, 15
; CHECK: lhi %r2, 0
; CHECK: br %r14
entry:
  %cmp = icmp sgt i32 %a, 14
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Check conditional compare grande immediate and trap
define i64 @f2(i64 signext %a) {
; CHECK-LABEL: f2:
; CHECK: cgitle %r2, 14
; CHECK: lghi %r2, 0
; CHECK: br %r14
entry:
  %cmp = icmp slt i64 %a, 15
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 0
}

; Check conditional compare logical immediate and trap
define i32 @f3(i32 zeroext %a) {
; CHECK-LABEL: f3:
; CHECK: clfithe %r2, 15
; CHECK: lhi %r2, 0
; CHECK: br %r14
entry:
  %cmp = icmp ugt i32 %a, 14
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Check conditional compare grande logical immediate and trap
define i64 @f4(i64 zeroext %a) {
; CHECK-LABEL: f4:
; CHECK: clgitle %r2, 14
; CHECK: lghi %r2, 0
; CHECK: br %r14
entry:
  %cmp = icmp ult i64 %a, 15
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 0
}

; Check conditional compare and trap
define i32 @f5(i32 signext %a, i32 signext %b) {
; CHECK-LABEL: f5:
; CHECK: crte %r2, %r3
; CHECK: lhi %r2, 0
; CHECK: br %r14
entry:
  %cmp = icmp eq i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Check conditional compare grande and trap
define i64 @f6(i64 signext %a, i64 signext %b) {
; CHECK-LABEL: f6:
; CHECK: cgrtl %r2, %r3
; CHECK: lghi %r2, 0
; CHECK: br %r14
entry:
  %cmp = icmp slt i64 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 0
}

; Check conditional compare logical and trap
define i32 @f7(i32 zeroext %a, i32 zeroext %b) {
; CHECK-LABEL: f7:
; CHECK: clrth %r2, %r3
; CHECK: lhi %r2, 0
; CHECK: br %r14
entry:
  %cmp = icmp ugt i32 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i32 0
}

; Check conditional compare logical grande and trap
define i64 @f8(i64 zeroext %a, i64 zeroext %b) {
; CHECK-LABEL: f8:
; CHECK: clgrtl %r2, %r3
; CHECK: lghi %r2, 0
; CHECK: br %r14
entry:
  %cmp = icmp ult i64 %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret i64 0
}

; Check conditional traps that don't have a valid Compare and Trap
define double @f9(double %a, double %b) {
; CHECK-LABEL: f9:
; CHECK: cdbr %f0, %f2
; CHECK-LABEL: .Ltmp1
; CHECK: je .Ltmp1+2
; CHECK: lzdr %f0
; CHECK: br %r14
entry:
  %cmp = fcmp oeq double %a, %b
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  tail call void @llvm.trap()
  unreachable

if.end:                                           ; preds = %entry
  ret double 0.000000e+00
}
