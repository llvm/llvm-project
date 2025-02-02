; RUN: llc < %s -mtriple=armv6-apple-ios0.0.0 -mcpu=arm1136jf-s -verify-machineinstrs | FileCheck %s
; RUN: llc -mtriple=thumbv6m-eabi -mattr=+execute-only -verify-machineinstrs %s -o - | FileCheck --check-prefix=CHECK-T1 %s
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:32:64-v128:32:128-a0:0:32-n32-S32"

; Don't CSE a cmp across a call that clobbers CPSR.
;
; CHECK: cmp

; CHECK-T1: movs [[REG:r[0-7]+]], :upper8_15:
; CHECK-T1-NEXT: lsls    [[REG]], [[REG]], #8
; CHECK-T1-NEXT: adds    [[REG]], :upper0_7:
; CHECK-T1-NEXT: lsls    [[REG]], [[REG]], #8
; CHECK-T1-NEXT: adds    [[REG]], :lower8_15:
; CHECK-T1-NEXT: lsls    [[REG]], [[REG]], #8
; CHECK-T1-NEXT: adds    [[REG]], :lower0_7:

; CHECK-T1: movs [[REG:r[0-7]+]], :upper8_15:
; CHECK-T1-NEXT: lsls    [[REG]], [[REG]], #8
; CHECK-T1-NEXT: adds    [[REG]], :upper0_7:
; CHECK-T1-NEXT: lsls    [[REG]], [[REG]], #8
; CHECK-T1-NEXT: adds    [[REG]], :lower8_15:
; CHECK-T1-NEXT: lsls    [[REG]], [[REG]], #8
; CHECK-T1-NEXT: adds    [[REG]], :lower0_7:

; CHECK-T1: cmp

; CHECK: S_trimzeros
; CHECK-T1: S_trimzeros
; CHECK-NOT: moveq
; CHECK-T1-NOT: beq
; CHECK: strlen

@F_floatmul.man1 = external global [200 x i8], align 1
@F_floatmul.man2 = external global [200 x i8], align 1

declare i32 @strlen(ptr nocapture) nounwind readonly
declare void @S_trimzeros(...)

define ptr @F_floatmul(ptr %f1, ptr %f2, i32 %a, i32 %b) nounwind ssp {
entry:
  %0 = icmp eq i32 %a, %b
  br i1 %0, label %while.end42, label %while.body37

while.body37:                                     ; preds = %while.body37, %entry
  br i1 false, label %while.end42, label %while.body37

while.end42:                                      ; preds = %while.body37, %entry
  %. = select i1 %0, ptr @F_floatmul.man1, ptr @F_floatmul.man2
  %.92 = select i1 %0, ptr @F_floatmul.man2, ptr @F_floatmul.man1
  tail call void @S_trimzeros(ptr %.92) nounwind
  %call47 = tail call i32 @strlen(ptr %.) nounwind
  unreachable
}
