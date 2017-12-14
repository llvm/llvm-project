; Test the case when the preferred (larger / more aligned) version of a common
; symbol is located in a module that's not included in the link.

; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/start-lib-common.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t1.o --start-lib %t2.o --end-lib -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
@x = common global i32 0, align 4

; ToT gold (as of 03/2016) honors --start-lib/--end-lib, drops %t2.o and ends up
; with (i32 align 4) symbol.
; Older gold does not drop %t2.o and ends up with (i32 align 8) symbol. This is
; incorrect behavior, but this test does not verify this in order to support
; both old and new gold.

; Check that the common symbol is not dropped completely, which was a regression
; in r262676.
; CHECK: @x = common global i32 0, align
