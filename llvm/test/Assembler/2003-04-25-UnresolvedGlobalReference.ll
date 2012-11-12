; RUN: llvm-as %s -o /dev/null
; There should be absolutely no problem with this testcase.

define i32 @test(i32 %arg1, i32 %arg2) {
        ret i32 ptrtoint (i32 (i32, i32)* @test to i32)
}

