; RUN: llc < %s -mattr +noabicalls -mgpopt | FileCheck %s -check-prefixes=MIPS64

target datalayout = "E-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "mips64-unknown-linux-muslabi64"

@.str = external constant [6 x i8]
@.str.1 = external constant [6 x i8]

define ptr @tst_select_ptr_ptr(i1 %tobool.not) {
entry:
; MIPS64-LABEL: tst_select_ptr_ptr:
; MIPS64:       # %bb.0: # %entry
; MIPS64:         sll $1, $4, 0
; MIPS64:         andi $1, $1, 1
; MIPS64:         daddiu $2, $gp, %gp_rel(.str)
; MIPS64:         daddiu $3, $gp, %gp_rel(.str.1)
; MIPS64:         jr $ra
; MIPS64:         movn $2, $3, $1

  %cond = select i1 %tobool.not, ptr @.str.1, ptr @.str
  ret ptr %cond
}
