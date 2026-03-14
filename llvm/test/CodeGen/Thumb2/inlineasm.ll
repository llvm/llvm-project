; RUN: llc < %s -filetype=obj

target triple = "thumbv7-none--eabi"

define void @t1() nounwind {
entry:
  call void asm sideeffect "mov r0, r1", ""() nounwind
  ret void
}
