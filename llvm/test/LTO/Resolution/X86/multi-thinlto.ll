; RUN: opt -module-summary %s -o %t.o
; RUN: llvm-cat -b -o %t2.o %t.o %t.o
; RUN: not llvm-lto2 run -o %t3.o %t2.o 2>&1 | FileCheck %s
; CHECK: Expected at most one ThinLTO module per bitcode file

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
