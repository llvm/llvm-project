; REQUIRES: x86-registered-target
; Compile with thinlto indices, to enable thinlto.
; RUN: opt -module-summary %s -o %t1.bc

; Test old lto interface with thinlto.
; RUN: llvm-lto -exported-symbol=main -thinlto-action=run %t1.bc
; RUN: llvm-nm %t1.bc.thinlto.o | FileCheck %s --check-prefix=CHECK-NM

; Test new lto interface with thinlto.
; RUN: llvm-lto2 run %t1.bc -o %t.out -save-temps \
; RUN:   -r %t1.bc,bar,pl \
; RUN:   -r %t1.bc,__stack_chk_guard,pl \
; RUN:   -r %t1.bc,__stack_chk_fail,pl
; RUN: llvm-nm %t.out.1 | FileCheck %s --check-prefix=CHECK-NM

; Re-compile, this time without the thinlto indices.
; RUN: opt %s -o %t4.bc

; Test the new lto interface without thinlto.
; RUN: llvm-lto2 run %t4.bc -o %t5.out -save-temps \
; RUN:   -r %t4.bc,bar,pl \
; RUN:   -r %t4.bc,__stack_chk_guard,pl \
; RUN:   -r %t4.bc,__stack_chk_fail,pl
; RUN: llvm-nm %t5.out.0 | FileCheck %s --check-prefix=CHECK-NM

; Test the old lto interface without thinlto.
; RUN: llvm-lto -exported-symbol=main %t4.bc -o %t6
; RUN: llvm-nm %t6 | FileCheck %s --check-prefix=CHECK-NM

; CHECK-NM-NOT: bar
; CHECK-NM: T __stack_chk_fail
; CHECK-NM: D __stack_chk_guard
; CHECK-NM-NOT: bar

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define void @bar() {
    ret void
}

@__stack_chk_guard = dso_local global i64 1, align 8

define void @__stack_chk_fail() {
    ret void
}
