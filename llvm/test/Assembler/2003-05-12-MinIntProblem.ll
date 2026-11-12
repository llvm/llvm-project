; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: i32 -2147483648
define i32 @foo() {
        ret i32 -2147483648
}
