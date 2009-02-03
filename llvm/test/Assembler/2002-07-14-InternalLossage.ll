; Test to make sure that the 'internal' tag is not lost!
;
; RUN: llvm-as < %s | llvm-dis | grep internal

declare void @foo()

define internal void @foo() {
        ret void
}
