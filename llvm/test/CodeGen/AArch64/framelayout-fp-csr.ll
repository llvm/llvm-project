; RUN: llc -verify-machineinstrs -mtriple=aarch64-none-linux-gnu -disable-post-ra --frame-pointer=all < %s | FileCheck %s

; The purpose of this test is to verify that frame pointer (x29)
; is correctly setup in the presence of callee-saved floating
; point registers.  The frame pointer should point to the frame
; record, which is located 16 bytes above the end of the CSR
; space when a single FP CSR is in use.
define void @test1(i32) #26 {
entry:
  call void asm sideeffect "nop", "~{d8}"() #26
  ret void
}
; CHECK-LABEL: test1:
; CHECK:       str     d8, [sp, #-32]!
; CHECK-NEXT:  stp     x29, x30, [sp, #16]
; CHECK-NEXT:  add     x29, sp, #16
; CHECK:       nop
; CHECK:       ldp     x29, x30, [sp, #16]
; CHECK-NEXT:  ldr     d8, [sp], #32
; CHECK-NEXT:  ret

attributes #26 = { nounwind }
