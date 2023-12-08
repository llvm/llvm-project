; RUN: llvm-as %s -o %t.o
; RUN: llvm-cat -b -o %t2.o %t.o %t.o
; RUN: llvm-dis -o %t3 %t2.o
; RUN: llvm-as -o %t4.o %t3.0
; RUN: llvm-as -o %t5.o %t3.1
; RUN: cmp %t4.o %t5.o
; RUN: llvm-cat -b -o %t6.o %t5.o %t4.o
; RUN: llvm-dis -o %t7.o %t6.o
; RUN: diff %t7.o.0 %t7.o.1
; RUN: FileCheck < %t7.o.0 %s
; RUN: FileCheck < %t7.o.1 %s

; CHECK: source_filename = "{{.*}}multi-mod-disassemble.ll"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
