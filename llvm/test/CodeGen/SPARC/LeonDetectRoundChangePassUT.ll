; RUN: llc %s -O0 -mtriple=sparc -mcpu=leon3 -mattr=+detectroundchange -o - 2>&1 | grep "detect rounding changes"

; Function Attrs: nounwind
declare i32 @fesetround(i32)

define void @test_round_change() {
entry:
  %call = call i32 @fesetround(i32 2048)

  ret void
}
