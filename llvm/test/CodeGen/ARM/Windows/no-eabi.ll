; RUN: llc -O3 -mtriple thumbv7-windows %s -filetype asm -o - | FileCheck -check-prefix CHECK-NONEABI %s
; RUN: llc -O3 -mtriple armv7--linux-gnueabi %s -filetype asm -o - | FileCheck -check-prefix CHECK-EABI %s

define arm_aapcs_vfpcc void @function() {
  ret void
}

; CHECK-EABI: .eabi_attribute
; CHECK-NONEABI-NOT: .eabi_attribute

