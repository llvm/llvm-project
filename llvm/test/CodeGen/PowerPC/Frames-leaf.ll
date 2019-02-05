; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   not grep "stw r31, 20(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   not grep "stwu r1, -.*(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   not grep "addi r1, r1, "
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- | \
; RUN:   not grep "lwz r31, 20(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -frame-pointer=all | \
; RUN:   not grep "stw r31, 20(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -frame-pointer=all | \
; RUN:   not grep "stwu r1, -.*(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -frame-pointer=all | \
; RUN:   not grep "addi r1, r1, "
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc32-- -frame-pointer=all | \
; RUN:   not grep "lwz r31, 20(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- | \
; RUN:   not grep "std r31, 40(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- | \
; RUN:   not grep "stdu r1, -.*(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- | \
; RUN:   not grep "addi r1, r1, "
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- | \
; RUN:   not grep "ld r31, 40(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -frame-pointer=all | \
; RUN:   not grep "stw r31, 40(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -frame-pointer=all | \
; RUN:   not grep "stdu r1, -.*(r1)"
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -frame-pointer=all | \
; RUN:   not grep "addi r1, r1, "
; RUN: llc -verify-machineinstrs < %s -mtriple=ppc64-- -frame-pointer=all | \
; RUN:   not grep "ld r31, 40(r1)"

define i32* @f1() {
        %tmp = alloca i32, i32 2                ; <i32*> [#uses=1]
        ret i32* %tmp
}
