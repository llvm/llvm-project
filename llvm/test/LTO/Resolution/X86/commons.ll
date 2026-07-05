; RUN: llvm-as -o %t1.bc %s
; RUN: llvm-as -o %t2.bc %p/Inputs/commons.ll
; RUN: llvm-lto2 run %t1.bc -r=%t1.bc,x,l %t2.bc -r=%t2.bc,x,pl -o %t.out -save-temps
; RUN: llvm-dis -o - %t.out.0.0.preopt.bc  | FileCheck %s

; A strong definition should override the common
; CHECK: @x = dso_local global i32 42, align 4

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@x = common global i16 0, align 2
