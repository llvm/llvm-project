; RUN: llc < %s -mtriple=mips   -mattr +noabicalls -mgpopt | FileCheck %s -check-prefixes=MIPS32
; RUN: llc < %s -mtriple=mips64 -mattr +noabicalls -mgpopt | FileCheck %s -check-prefixes=MIPS64

@.str = external constant [6 x i8]
@.str.1 = external constant [6 x i8]

define ptr @tst_select_ptr_ptr(i1 %tobool.not) {
; MIPS32-LABEL: tst_select_ptr_ptr:
; MIPS32:       # %bb.0: # %entry
; MIPS32-NEXT:    andi $[[R0:[0-9]+]], $4, 1
; MIPS32-NEXT:    addiu $[[T0:[0-9]+]], $gp, %gp_rel(.str)
; MIPS32-NEXT:    addiu $[[T1:[0-9]+]], $gp, %gp_rel(.str.1)
; MIPS32-NEXT:    jr $ra
; MIPS32-NEXT:    movn $[[T0]], $[[T1]], $[[R0]]

; MIPS64-LABEL: tst_select_ptr_ptr:
; MIPS64:       # %bb.0: # %entry
; MIPS64-NEXT:    sll $[[R0:[0-9]+]], $4, 0
; MIPS64-NEXT:    andi $[[R0]], $[[R0]], 1
; MIPS64-NEXT:    daddiu $[[T0:[0-9]+]], $gp, %gp_rel(.str)
; MIPS64-NEXT:    daddiu $[[T1:[0-9]+]], $gp, %gp_rel(.str.1)
; MIPS64-NEXT:    jr $ra
; MIPS64-NEXT:    movn $[[T0]], $[[T1]], $[[R0]]

entry:
  %cond = select i1 %tobool.not, ptr @.str.1, ptr @.str
  ret ptr %cond
}
