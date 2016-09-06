; Verify that auto-upgrading intrinsics works with Lazy loaded bitcode
; Do setup work for all below tests: generate bitcode and combined index
; RUN: opt -module-summary %s -o %t.bc
; RUN: llvm-lto -thinlto-action=thinlink -o %t3.bc %t.bc %p/Inputs/autoupgrade.bc

; RUN: llvm-lto -thinlto-action=import %t.bc -thinlto-index=%t3.bc -o - | llvm-bcanalyzer -dump | FileCheck %s

; We can't use llvm-dis here, because it would do the autoupgrade itself.

; CHECK-NOT: 'llvm.invariant.start'
; CHECK: record string = 'llvm.invariant.start.p0i8'
; CHECK-NOT: 'llvm.invariant.start'

target datalayout = "e-m:o-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-apple-macosx10.11.0"

define i32 @main() #0 {
entry:
  call void (...) @globalfunc1()
  ret i32 0
}

declare void @globalfunc1(...) #1
