; RUN: llc -filetype=null < %s

target triple = "hexagon"

define void @foo() {
  tail call void asm sideeffect "//", ""()
  ret void
}
