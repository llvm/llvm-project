; RUN: opt -module-summary %s -o %t.bc
; RUN: opt -module-summary %p/Inputs/funcimport_appending_global_used.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t.bc %t2.bc

; Do the import now
; RUN: llvm-link %t.bc -summary-index=%t3.thinlto.bc -import=foo:%t2.bc -S | FileCheck %s

; Test case where the verifier would fail if checking use_empty
; instead of materialized_use_empty on llvm.used.

; CHECK: @llvm.used = appending global [1 x ptr] [ptr @f]

declare void @f()
@llvm.used = appending global [1 x ptr] [ptr @f]

define i32 @main() {
entry:
  call void @foo()
  ret i32 0
}

declare void @foo()
