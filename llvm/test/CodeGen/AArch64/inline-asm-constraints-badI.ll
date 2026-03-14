; RUN: not llc -mtriple=aarch64-none-linux-gnu -o - %s

define void @foo() {
  ; Out of range immediate for I.
  call void asm sideeffect "add x0, x0, $0", "I"(i32 4097)
  ret void
}
