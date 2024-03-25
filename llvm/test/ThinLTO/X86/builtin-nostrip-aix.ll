; REQUIRES: powerpc-registered-target
; Compile with thinlto indices, to enable thinlto.
; RUN: opt -module-summary %s -o %t1.bc

; Test old lto interface with thinlto.
; RUN: llvm-lto -exported-symbol=main -thinlto-action=run %t1.bc
; RUN: llvm-nm %t1.bc | FileCheck %s --check-prefix=CHECK-NM

; Test new lto interface with thinlto.
; RUN: llvm-lto2 run %t1.bc -o %t.out -save-temps \
; RUN:   -r %t1.bc,bar,pl \
; RUN:   -r %t1.bc,__ssp_canary_word,pl \
; RUN:   -r %t1.bc,__stack_chk_fail,pl
; RUN: llvm-nm %t.out.1.2.internalize.bc | FileCheck %s --check-prefix=CHECK-NM

; Re-compile, this time without the thinlto indices.
; RUN: opt %s -o %t4.bc

; Test the new lto interface without thinlto.
; RUN: llvm-lto2 run %t4.bc -o %t5.out -save-temps \
; RUN:   -r %t4.bc,bar,pl \
; RUN:   -r %t4.bc,__ssp_canary_word,pl \
; RUN:   -r %t4.bc,__stack_chk_fail,pl
; RUN: llvm-nm %t5.out.0.2.internalize.bc | FileCheck %s --check-prefix=CHECK-NM

; Test the old lto interface without thinlto.
; RUN: llvm-lto -exported-symbol=main -save-merged-module %t4.bc -o %t6
; RUN: llvm-nm %t6.merged.bc | FileCheck %s --check-prefix=CHECK-NM

; CHECK-NM: D __ssp_canary_word
; CHECK-NM: T __stack_chk_fail

target datalayout = "E-m:a-p:32:32-Fi32-i64:64-n32"
target triple = "powerpc-ibm-aix-xcoff"

define void @bar() {
    ret void
}

@__ssp_canary_word = dso_local global i64 1, align 8

define void @__stack_chk_fail() {
    ret void
}
