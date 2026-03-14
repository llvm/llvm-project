; RUN: not llc -mtriple=aarch64-none-linux-gnu -o - %s

define void @foo() {
  ; 32-bit bitpattern ending in 1101 can't be produced.
  call void asm sideeffect "and w0, w0, $0", "K"(i64 4294967296)
  ret void
}
