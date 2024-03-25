; REQUIRES: asserts
; Test to ensure that the Unified LTO flag is set properly in the summary, and
; that we emit the correct error when linking bitcode files with different
; values of this flag.

; Linking bitcode both without UnifiedLTO set should work
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t1 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck %s --check-prefix=NOUNIFIEDLTO
; RUN: llvm-dis -o - %t1 | FileCheck %s --check-prefix=NOUNIFIEDLTOFLAG
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t2 %s
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck %s --check-prefix=NOUNIFIEDLTO
; RUN: llvm-dis -o - %t2 | FileCheck %s --check-prefix=NOUNIFIEDLTOFLAG
; RUN: llvm-lto2 run -o %t3 %t1 %t2
; RUN: not llvm-lto2 run --unified-lto=thin -o %t3 %t1 %t2 2>&1 | \
; RUN: FileCheck --allow-empty %s --check-prefix UNIFIEDERR

; Linking bitcode with different values of UnifiedLTO should fail
; RUN: opt -thinlto-bc -thinlto-split-lto-unit -o %t1 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck %s --check-prefix=NOUNIFIEDLTO
; RUN: llvm-dis -o - %t1 | FileCheck %s --check-prefix=NOUNIFIEDLTOFLAG
; RUN: opt -unified-lto -thinlto-bc -thinlto-split-lto-unit -o %t2 %s
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck %s --check-prefix=UNIFIEDLTO
; RUN: llvm-dis -o - %t2 | FileCheck %s --check-prefix=UNIFIEDLTOFLAG
; RUN: not llvm-lto2 run --unified-lto=thin -o %t3 %t1 %t2 2>&1 | \
; RUN: FileCheck --allow-empty %s --check-prefix UNIFIEDERR

; Linking bitcode with identical Unified LTO flags should succeed
; RUN: opt -unified-lto -thinlto-bc -thinlto-split-lto-unit -o %t1 %s
; RUN: llvm-bcanalyzer -dump %t1 | FileCheck %s --check-prefix=UNIFIEDLTO
; RUN: llvm-dis -o - %t1 | FileCheck %s --check-prefix=UNIFIEDLTOFLAG
; RUN: opt -unified-lto -thinlto-bc -thinlto-split-lto-unit -o %t2 %s
; RUN: llvm-bcanalyzer -dump %t2 | FileCheck %s --check-prefix=UNIFIEDLTO
; RUN: llvm-dis -o - %t2 | FileCheck %s --check-prefix=UNIFIEDLTOFLAG
; RUN: llvm-lto2 run --unified-lto=full --debug-only=lto -o %t3 %t1 %t2 2>&1 | \
; RUN: FileCheck --allow-empty %s --check-prefix NOUNIFIEDERR --check-prefix FULL
; RUN: llvm-lto2 run --unified-lto=thin --debug-only=lto -o %t3 %t1 %t2 2>&1 | \
; RUN: FileCheck --allow-empty %s --check-prefix NOUNIFIEDERR --check-prefix THIN
; RUN: llvm-lto2 run --debug-only=lto -o %t3 %t1 %t2 2>&1 | \
; RUN: FileCheck --allow-empty %s --check-prefix THIN

; UNIFIEDERR: unified LTO compilation must use compatible bitcode modules
; NOUNIFIEDERR-NOT: unified LTO compilation must use compatible bitcode modules

; The flag should be set when UnifiedLTO is enabled
; UNIFIEDLTO: <FLAGS op0=520/>
; NOUNIFIEDLTO: <FLAGS op0=8/>

; Check that the corresponding module flag is set when expected.
; UNIFIEDLTOFLAG: !{i32 1, !"UnifiedLTO", i32 1}
; NOUNIFIEDLTOFLAG-NOT: !{i32 1, !"UnifiedLTO", i32 1}

; FULL: Running regular LTO
; THIN: Running ThinLTO

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
