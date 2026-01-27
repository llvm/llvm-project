; RUN: llc < %s -mattr +noabicalls -mgpopt | FileCheck %s -check-prefixes=MIPS64

target triple = "mips64-unknown-linux-muslabi64"

@.str = external constant [6 x i8]
@.str.1 = external constant [6 x i8]

define ptr @tst_select_ptr_ptr(i1 %tobool.not) {
; MIPS64-LABEL: tst_select_ptr_ptr:
; MIPS64:       # %bb.0: # %entry
; MIPS64-NEXT:    sll $1, $4, 0
; MIPS64-NEXT:    andi $1, $1, 1
; MIPS64-NEXT:    daddiu $2, $zero, %gp_rel(.str)
; MIPS64-NEXT:    daddiu $3, $zero, %gp_rel(.str.1)
; MIPS64-NEXT:    movn $2, $3, $1
; MIPS64-NEXT:    jr $ra
; MIPS64-NEXT:    daddu $2, $gp, $2
entry:
  %cond = select i1 %tobool.not, ptr @.str.1, ptr @.str
  ret ptr %cond
}
