; RUN: not llc -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8a -mattr=+altivec %s -o - 2>&1 | FileCheck %s

define hidden void @f(i32 %x) {
  ; CHECK: scalar-to-vector conversion failed, possible invalid constraint for vector type
  tail call void asm sideeffect "nop", "{v1}"(i32 %x) nounwind

  ; CHECK: scalar-to-vector conversion failed, possible invalid constraint for vector type
  tail call void asm sideeffect "nop", "{vsl1}"(i32 %x) nounwind

  ; CHECK: scalar-to-vector conversion failed, possible invalid constraint for vector type
  tail call void asm sideeffect "nop", "{vsh1}"(i32 %x) nounwind

  ret void
}
