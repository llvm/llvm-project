; RUN: llvm-as < %s | llvm-dis | grep -- -2147483648

define i32 @foo() {
        ret i32 -2147483648
}
