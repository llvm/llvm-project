; RUN: llvm-as %s -disable-output
; RUN: verify-uselistorder %s

define void @test(i32 %X) {
        call void @test( i32 6 )
        ret void
}
