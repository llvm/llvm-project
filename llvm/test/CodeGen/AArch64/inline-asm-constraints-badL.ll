; RUN: not llc -mtriple=arm64-apple-ios7.0 -o - %s

define void @foo() {
  ; 32-bit bitpattern ending in 1101 can't be produced.
  call void asm sideeffect "and x0, x0, $0", "L"(i32 13)
  ret void
}
