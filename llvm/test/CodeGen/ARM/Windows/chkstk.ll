; RUN: llc -mtriple=thumbv7-windows -mcpu=cortex-a9 -verify-machineinstrs %s -o - \
; RUN:  | FileCheck -check-prefix CHECK-DEFAULT-CODE-MODEL %s

; RUN: llc -mtriple=thumbv7-windows -mcpu=cortex-a9 -code-model=large -verify-machineinstrs %s -o - \
; RUN:  | FileCheck -check-prefix CHECK-LARGE-CODE-MODEL %s

define arm_aapcs_vfpcc void @check_watermark() {
entry:
  %buffer = alloca [4096 x i8], align 1
  ret void
}

; CHECK-DEFAULT-CODE-MODEL: check_watermark:
; CHECK-DEFAULT-CODE-MODEL: 	movw r4, #1024
; CHECK-DEFAULT-CODE-MODEL: 	bl __chkstk
; CHECK-DEFAULT-CODE-MODEL: 	sub.w sp, sp, r4

; CHECK-LARGE-CODE-MODEL: check_watermark:
; CHECK-LARGE-CODE-MODEL: 	movw r12, :lower16:__chkstk
; CHECK-LARGE-CODE-MODEL: 	movt r12, :upper16:__chkstk
; CHECK-LARGE-CODE-MODEL: 	movw r4, #1024
; CHECK-LARGE-CODE-MODEL: 	blx r12
; CHECK-LARGE-CODE-MODEL: 	sub.w sp, sp, r4

