; RUN: llc -mtriple=arm-eabi -mattr=+vfp2 %s -o /dev/null

define i64 @test_i64(i64 %a1, i64 %a2) {
        ret i64 %a2
}

