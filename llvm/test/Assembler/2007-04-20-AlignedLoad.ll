; RUN: llvm-as < %s | llvm-dis | FileCheck %s
; RUN: verify-uselistorder %s

; CHECK: %tmp2 = load i32, ptr %arg, align 1024
define i32 @test(ptr %arg) {
entry:
        %tmp2 = load i32, ptr %arg, align 1024      ; <i32> [#uses=1]
        ret i32 %tmp2
}
