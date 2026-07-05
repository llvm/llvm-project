; RUN: llc -mtriple=arm-eabi %s -o /dev/null

define i32 @test(i32 %a1, i32 %a2) {
        ret i32 %a2
}

