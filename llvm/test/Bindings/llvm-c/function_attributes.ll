; RUN: llvm-as %s -o %t.bc
; RUN: llvm-c-test --test-function-attributes < %t.bc
; This used to segfault

define void @X() {
    ret void
}
