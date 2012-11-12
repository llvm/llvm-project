; RUN: llc < %s -march=x86 | grep -i ESP | not grep sub

define i32 @test(i32 %X) {
        ret i32 %X
}
