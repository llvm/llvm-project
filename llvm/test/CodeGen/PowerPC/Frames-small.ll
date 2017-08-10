; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin8 -o %t1
; RUN: not grep "stw r31, -4(r1)" %t1
; RUN: grep "stwu r1, -16416(r1)" %t1
; RUN: grep "addi r1, r1, 16416" %t1
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN: not grep "lwz r31, -4(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc-apple-darwin8 -disable-fp-elim \
; RUN:   -o %t2
; RUN: grep "stw r31, -4(r1)" %t2
; RUN: grep "stwu r1, -16416(r1)" %t2
; RUN: grep "addi r1, r1, 16416" %t2
; RUN: grep "lwz r31, -4(r1)" %t2
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin8 -o %t3
; RUN: not grep "std r31, -8(r1)" %t3
; RUN: grep "stdu r1, -16432(r1)" %t3
; RUN: grep "addi r1, r1, 16432" %t3
; RUN: not grep "ld r31, -8(r1)" %t3
; RUN: llc -verify-machineinstrs < %s -mtriple=powerpc64-apple-darwin8 -disable-fp-elim \
; RUN:   -o %t4
; RUN: grep "std r31, -8(r1)" %t4
; RUN: grep "stdu r1, -16448(r1)" %t4
; RUN: grep "addi r1, r1, 16448" %t4
; RUN: grep "ld r31, -8(r1)" %t4

define i32* @f1() {
        %tmp = alloca i32, i32 4095             ; <i32*> [#uses=1]
        ret i32* %tmp
}

