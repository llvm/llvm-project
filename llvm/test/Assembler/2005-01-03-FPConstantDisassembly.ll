; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

define double @test() {
; CHECK: ret double 1.0
        ret double 1.0   ;; This should not require hex notation
}

