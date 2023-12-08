; RUN: llvm-as %s -o %t.bc
; RUN: llvm-c-test --test-callsite-attributes < %t.bc
; This used to segfault

define void @Y() {
    ret void
}

define void @X() {
    call void @X()
    ret void
}
