; One file may have multiple functions targeted at different (ARM, Thumb)
; instruction sets. Passing this information to the linker and the assembler
; is done through the ".code 16" and ".code 32" directives.
;
; RUN: llc -mtriple=arm-arm-none-eabi %s -o - | FileCheck %s

define void @ft() #0 {
; CHECK: .code 16
; CHECK: .thumb_func
; CHECK-LABEL: ft:
entry:
  ret void
}

define void @fz() #1 {
; CHECK: .code 32
; CHECK-LABEL: fz:
entry:
  ret void
}

attributes #0 = { "target-features"="+thumb-mode" }
attributes #1 = { "target-features"="-thumb-mode" }
