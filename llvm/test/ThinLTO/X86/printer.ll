; RUN: llvm-as %s -o %t.bc
; RUN: llvm-lto2 run -r=%t.bc,foo,pxl --print-before-all %t.bc -o %t2 2>&1 | FileCheck %s --check-prefix=CHECK-BEFORE
; RUN: llvm-lto2 run -r=%t.bc,foo,pxl --print-after-all %t.bc -o %t3 2>&1 | FileCheck %s --check-prefix=CHECK-AFTER
; CHECK-BEFORE: *** IR Dump Before GlobalDCEPass
; CHECK-AFTER: *** IR Dump After GlobalDCEPass

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

define i32 @foo() {
    ret i32 42
}
