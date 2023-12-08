; RUN: opt -module-summary %s -o %t1.bc
; RUN: llvm-lto2 run %t1.bc -o %t.o \
; RUN:     -r=%t1.bc,_tinkywinky,pxl \
; RUN:     -debug-pass-manager \
; RUN:     2>&1 | FileCheck %s

; RUN: llvm-lto -thinlto-action=run -exported-symbol tinkywinky \
; RUN:          -debug-pass-manager \
; RUN:          %t1.bc 2>&1 | FileCheck %s

; Check that the selected pass manager is used for middle-end optimizations by
; checking the debug output for IPSCCP.

; CHECK-NOT: Interprocedural Sparse Conditional Constant Propagation
; CHECK: Running pass: IPSCCPPass on [module]
; CHECK-NOT: Interprocedural Sparse Conditional Constant Propagation

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @tinkywinky() {
    ret void
}
