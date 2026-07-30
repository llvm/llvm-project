; RUN: llvm-as %s -o %t.o
; RUN: %gold -shared -m elf_x86_64 -o %t2 -plugin %llvmshlibdir/LLVMgold%shlibext %t.o
; RUN: llvm-nm %t2 | FileCheck %s
; CHECK: PrepareAndDispatch

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm "call PrepareAndDispatch@plt"
module asm "\09"
