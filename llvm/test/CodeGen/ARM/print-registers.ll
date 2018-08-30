; RUN: llc -mtriple=armeb-arm-none-eabi < %s -o -| FileCheck %s -check-prefixes=CHECK-BE
; RUN: llc -mtriple=arm-arm-none-eabi < %s -o -| FileCheck %s -check-prefixes=CHECK-LE

define dso_local void @_Z3fooi(i32 %a) local_unnamed_addr #0 {
entry:
; CHECK-BE: @ plain: [[LOW_REG:r[0-9]+]] Q: [[HIGH_REG:r[0-9]+]] R: [[LOW_REG]] H: [[HIGH_REG]]
; CHECK-LE: @ plain: [[LOW_REG:r[0-9]+]] Q: [[LOW_REG]] R: [[HIGH_REG:r[0-9]+]] H: [[HIGH_REG]]
  tail call void asm sideeffect "// plain: $0 Q: ${0:Q} R: ${0:R} H: ${0:H}", "r"(i64 1) #1
  ret void
}
