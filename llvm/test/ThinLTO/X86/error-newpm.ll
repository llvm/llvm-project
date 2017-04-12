; RUN: opt -module-summary %s -o %t1.bc
; RUN: not llvm-lto2 run %t1.bc -o %t.o \
; RUN:     -r=%t1.bc,_tinkywinky,pxl \
; RUN:     -lto-use-new-pm 2>&1 | FileCheck %s

; CHECK: ThinLTO not supported with the new PM yet!

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define void @tinkywinky() {
    ret void
}
