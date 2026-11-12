; Test to make sure dot dumper can correctly handle aliases to multiple
; different internal aliasees with the same name.

; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/alias_internal.ll -o %t2.bc
; RUN: llvm-lto2 run %t1.bc %t2.bc -o %t.out -save-temps \
; RUN:   -r %t1.bc,a1,plx \
; RUN:   -r %t2.bc,a2,plx

; RUN: cat %t.out.index.dot | FileCheck %s
; CHECK-DAG: M0_12511626713252727690 -> M0_{{.*}} // alias
; CHECK-DAG: M1_8129049334585965161 -> M1_{{.*}} // alias

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define internal i32 @f(ptr) unnamed_addr {
    ret i32 42
}

@a1 = weak alias i32 (ptr), ptr @f
