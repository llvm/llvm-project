; Check that a "target-abi" module flag conflicting with the
; -target-abi command-line option is diagnosed once.
; RUN: not llc -mtriple=armv7-none-eabi -target-abi=aapcs -filetype=null < %s 2>&1 \
; RUN:   | FileCheck %s -implicit-check-not=error:

; CHECK: error: -target-abi option != target-abi module flag
define float @f1(float %x) #0 {
  %r = fadd float %x, %x
  ret float %r
}

define float @f2(float %x) #1 {
  %r = fadd float %x, %x
  ret float %r
}

attributes #0 = { "target-cpu"="cortex-a8" }
attributes #1 = { "target-cpu"="cortex-a15" }

!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"apcs"}
