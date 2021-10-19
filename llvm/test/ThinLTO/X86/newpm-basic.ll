; RUN: opt -module-summary %s -o %t1.bc
; RUN: llvm-lto2 run %t1.bc -o %t.o \
; RUN:     -r=%t1.bc,_tinkywinky,pxl \
; RUN:     -debug-pass-manager \
; RUN:     -use-new-pm 2>&1 | FileCheck --check-prefix=NEWPM %s

; RUN: llvm-lto -thinlto-action=run -exported-symbol tinkywinky \
; RUN:          -use-new-pm \
; RUN:          -debug-pass-manager \
; RUN:          %t1.bc 2>&1 | FileCheck --check-prefix=NEWPM %s

; RUN: llvm-lto -thinlto-action=run -exported-symbol tinkywinky \
; RUN:          -use-new-pm=false \
; RUN:          -debug-pass-manager \
; RUN:          -debug-pass=Structure \
; RUN:          %t1.bc 2>&1 | FileCheck --check-prefix=LEGACYPM %s

; Check that the selected pass manager is used for middle-end optimizations by
; checking the debug output for IPSCCP.

; NEWPM-NOT: Interprocedural Sparse Conditional Constant Propagation
; NEWPM: Running pass: IPSCCPPass on [module]
; NEWPM-NOT: Interprocedural Sparse Conditional Constant Propagation

; LEGACYPM-NOT: IPSCCPPass
; LEGACYPM:  Interprocedural Sparse Conditional Constant Propagation
; LEGACYPM-NOT: IPSCCPPass

target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @tinkywinky() {
    ret void
}
