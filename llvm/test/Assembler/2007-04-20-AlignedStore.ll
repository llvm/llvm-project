; RUN: llvm-as < %s | llvm-dis | grep "align 1024"
; RUN: verify-uselistorder %s

define void @test(ptr %arg) {
entry:
        store i32 0, ptr %arg, align 1024
        ret void
}
