; RUN: opt -module-summary %s -o %t1.bc
; RUN: opt -module-summary %p/Inputs/funcimport2.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t1.bc %t2.bc
; RUN: llvm-link -import=bar:%t2.bc %t1.bc -summary-index=%t3.thinlto.bc -S | FileCheck %s

; CHECK: define available_externally hidden void @foo() {
define available_externally hidden void @foo() {
    ret void
}

declare void @bar()

define void @caller() {
  call void @bar()
  call void @foo()
  ret void
}
