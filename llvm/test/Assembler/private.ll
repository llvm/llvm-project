; Test to make sure that the 'private' tag is not lost!
;
; RUN: llvm-as < %s | llvm-dis | grep private

declare void @foo()

define private void @foo() {
        ret void
}
