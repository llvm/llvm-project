; RUN: llvm-as < %s | llvm-dis | grep -- -2147483648
; RUN: verify-uselistorder %s

define i32 @foo() {
        ret i32 -2147483648
}
