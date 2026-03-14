; Test the case when the preferred (larger / more aligned) version of a common
; symbol is located in a module that's not included in the link.

; RUN: llvm-as %s -o %t1.o
; RUN: llvm-as %p/Inputs/start-lib-common.ll -o %t2.o

; RUN: %gold -plugin %llvmshlibdir/LLVMgold%shlibext \
; RUN:    --plugin-opt=emit-llvm \
; RUN:    -shared %t1.o --start-lib %t2.o --end-lib -o %t3.o
; RUN: llvm-dis %t3.o -o - | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = common global i32 0, align 4

; v1.12 gold honors --start-lib/--end-lib, drops %t2.o and ends up
; with (i32 align 4) symbol.
; CHECK: @x = common global i32 0, align 4
