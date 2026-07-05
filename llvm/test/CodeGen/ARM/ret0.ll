; RUN: llc -mtriple=arm-eabi %s -o /dev/null

define i32 @test() {
        ret i32 0
}
