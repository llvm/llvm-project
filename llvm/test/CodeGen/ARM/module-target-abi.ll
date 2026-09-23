; The "target-abi" module flag selects the ABI used for codegen. APCS uses
; 4-byte stack alignment while AAPCS uses 8-byte alignment, which is observable
; in the emitted prologue. The flag drives this with no -target-abi option.
; RUN: split-file %s %t
; RUN: llc -mtriple=armv7-none-eabi -filetype=asm < %t/apcs.ll | FileCheck %s --check-prefix=APCS
; RUN: llc -mtriple=armv7-none-eabi -filetype=asm < %t/aapcs.ll | FileCheck %s --check-prefix=AAPCS

;--- apcs.ll
; APCS: push {lr}
; APCS: sub sp, sp, #4
declare void @use(ptr)
define void @f() {
  %a = alloca i32
  call void @use(ptr %a)
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"apcs"}

;--- aapcs.ll
; AAPCS: push {r11, lr}
; AAPCS: sub sp, sp, #8
declare void @use(ptr)
define void @f() {
  %a = alloca i32
  call void @use(ptr %a)
  ret void
}
!llvm.module.flags = !{!0}
!0 = !{i32 1, !"target-abi", !"aapcs"}
