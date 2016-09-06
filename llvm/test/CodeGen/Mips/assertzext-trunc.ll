; RUN: llc < %s -march=mips64 -mcpu=mips3 | \
; RUN:   FileCheck %s -check-prefixes=ALL,PRE-R6
; RUN: llc < %s -march=mips64 -mcpu=mips4 | \
; RUN:   FileCheck %s -check-prefixes=ALL,PRE-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64 | \
; RUN:   FileCheck %s -check-prefixes=ALL,PRE-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r2 | \
; RUN:   FileCheck %s -check-prefixes=ALL,PRE-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r3 | \
; RUN:   FileCheck %s -check-prefixes=ALL,PRE-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r5 | \
; RUN:   FileCheck %s -check-prefixes=ALL,PRE-R6
; RUN: llc < %s -march=mips64 -mcpu=mips64r6 | \
; RUN:   FileCheck %s -check-prefixes=ALL,R6

; Check that we don't emit redundant SLLs for sequences of
; (AssertZext:i32 (trunc:i32 (AssertZext:i64 X, i32)), i8)
define zeroext i8 @udiv_i8(i8 zeroext %a, i8 zeroext %b) {
entry:
; ALL-LABEL: udiv_i8:

  ; PRE-R6-NOT:   sll     {{.*}}
  ; PRE-R6:       divu    $zero, $4, $5
  ; PRE-R6:       teq     $5, $zero, 7
  ; PRE-R6:       mflo    $2

  ; R6-NOT:       sll     {{.*}}
  ; R6:           divu    $2, $4, $5
  ; R6:           teq     $5, $zero, 7

  %r = udiv i8 %a, %b
  ret i8 %r
}

; Check that we do sign-extend when we have a (trunc:i32 (AssertZext:i64 X, i32))
define i64 @foo1(i64 zeroext %var) {
entry:
; ALL-LABEL: foo1:

  %shr = lshr i64 %var, 32
  %cmp = icmp eq i64 %shr, 0
  br i1 %cmp, label %if.end6, label %if.then

  ; ALL:    dsrl   $[[T0:[0-9]+]], $4, 32
  ; ALL:    sll    $[[T1:[0-9]+]], $[[T0]], 0
  if.then:                                          ; preds = %entry
  %conv = trunc i64 %shr to i32
  %cmp2 = icmp slt i32 %conv, 0
  br i1 %cmp2, label %if.then4, label %if.else

  if.then4:                                         ; preds = %if.then
  %add = add i64 %var, 16
  br label %if.end6

  if.else:                                          ; preds = %if.then
  %add5 = add i64 %var, 32
  br label %if.end6

  if.end6:                                          ; preds = %entry, %if.then4, %if.else
  %var.addr.0 = phi i64 [ %add, %if.then4 ], [ %add5, %if.else ], [ %var, %entry ]
  ret i64 %var.addr.0
}
