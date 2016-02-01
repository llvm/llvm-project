; RUN: llvm-as -function-summary %s -o %t1.bc
; RUN: llvm-as -function-summary %p/Inputs/funcimport2.ll -o %t2.bc
; RUN: llvm-lto -thinlto -o %t3 %t1.bc %t2.bc
; RUN: llvm-link -import=bar:%t2.bc %t1.bc -functionindex=%t3.thinlto.bc -S | FileCheck %s

; CHECK: define linkonce_odr hidden void @foo() {
define available_externally hidden void @foo() {
    ret void
}

declare void @bar()

define void @caller() {
  call void @bar()
  call void @foo()
  ret void
}
