; RUN: llvm-as < %s >%t1
; RUN: llvm-lto -o %t2 %t1
; REQUIRES: default_triple

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

module asm ".align 4"
