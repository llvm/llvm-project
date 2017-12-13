; RUN: llc -O2 -mtriple armv7a < %s | FileCheck --check-prefix=ARM %s

; RUN: llc -O2 -mtriple thumbv6m < %s | FileCheck --check-prefix=THUMB1 %s
; RUN: llc -O2 -mtriple thumbv8m.base < %s | FileCheck --check-prefix=THUMB1 %s

; RUN: llc -O2 -mtriple thumbv7a < %s | FileCheck --check-prefix=THUMB %s
; RUN: llc -O2 -mtriple thumbv8m.main < %s | FileCheck --check-prefix=THUMB %s

define void @fn1(i32 %a, i32 %b, i32 %c) local_unnamed_addr #0 {
entry:

; ARM: rsb	r2, r2, #1
; ARM: adds	r0, r1, r0
; ARM: movw	r1, #65535
; ARM: sxth	r2, r2
; ARM: adc	r0, r2, #0
; ARM: tst	r0, r1
; ARM: bxeq	lr
; ARM: .LBB0_1:
; ARM: b	.LBB0_1

; THUMB1: movs	r3, #1
; THUMB1: subs	r2, r3, r2
; THUMB1: sxth	r2, r2
; THUMB1: movs	r3, #0
; THUMB1: adds	r0, r1, r0
; THUMB1: adcs	r3, r2
; THUMB1: lsls	r0, r3, #16
; THUMB1: beq	.LBB0_2
; THUMB1: .LBB0_1:
; THUMB1: b	.LBB0_1

; THUMB: rsb.w	r2, r2, #1
; THUMB: adds	r0, r0, r1
; THUMB: sxth	r2, r2
; THUMB: adc	r0, r2, #0
; THUMB: lsls	r0, r0, #16
; THUMB: it	eq
; THUMB: bxeq	lr
; THUMB: .LBB0_1:
; THUMB: b	.LBB0_1

  %add = add i32 %b, %a
  %cmp = icmp ult i32 %add, %b
  %conv = zext i1 %cmp to i32
  %sub = sub i32 1, %c
  %add1 = add i32 %sub, %conv
  %conv2 = trunc i32 %add1 to i16
  %tobool = icmp eq i16 %conv2, 0
  br i1 %tobool, label %if.end, label %for.cond.preheader

for.cond.preheader:                               ; preds = %entry
  br label %for.cond

for.cond:                                         ; preds = %for.cond.preheader, %for.cond
  br label %for.cond

if.end:                                           ; preds = %entry
  ret void
}
