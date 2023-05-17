;; This test verifies llc on AIX would not crash when llvm.used and
;; llvm.compiler.used is presented in the IR.

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc-ibm-aix-xcoff -data-sections=false < %s | \
; RUN:   FileCheck %s

; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mtriple powerpc64-ibm-aix-xcoff -data-sections=false < %s | \
; RUN:   FileCheck %s

@keep_this = internal global i32 2, align 4
@keep_this2 = internal global i32 3, align 4
@llvm.used = appending global [1 x ptr] [ptr @keep_this], section "llvm.metadata"
@llvm.compiler.used = appending global [1 x ptr] [ptr @keep_this2], section "llvm.metadata"

; CHECK-NOT: llvm.metadata
; CHECK-NOT: llvm.used
; CHECK-NOT: llvm.compiler.used

; CHECK:    .lglobl keep_this
; CHECK:  keep_this:
; CHECK:    .lglobl keep_this2
; CHECK:  keep_this2:

; CHECK-NOT: llvm.metadata
; CHECK-NOT: llvm.used
; CHECK-NOT: llvm.compiler.used
