; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto -hot-cold-split=true -thinlto-action=run %t.bc -debug-pass=Structure 2>&1 | FileCheck %s -check-prefix=OLDPM-ANYLTO-POSTLINK-Os
; RUN: llvm-lto -hot-cold-split=true %t.bc -debug-pass=Structure 2>&1 | FileCheck %s -check-prefix=OLDPM-ANYLTO-POSTLINK-Os

; REQUIRES: asserts

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; OLDPM-ANYLTO-POSTLINK-Os: Hot Cold Splitting
