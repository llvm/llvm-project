; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: align 1024
define void @test(ptr %arg) {
entry:
        store i32 0, ptr %arg, align 1024
        ret void
}
