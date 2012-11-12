; RUN: llc < %s -march=thumb | not grep CPI


define i32 @test1() {
  ret i32 1000
}

define i32 @test2() {
  ret i32 -256
}
