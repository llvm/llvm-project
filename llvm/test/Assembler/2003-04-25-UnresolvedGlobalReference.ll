; RUN: llvm-as %s -o /dev/null
; RUN: verify-uselistorder %s
; There should be absolutely no problem with this testcase.

define i32 @test(i32 %arg1, i32 %arg2) {
        ret i32 ptrtoint (ptr @test to i32)
}

