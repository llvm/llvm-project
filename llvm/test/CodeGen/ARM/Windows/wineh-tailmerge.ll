;; Check that epilogues aren't tail merged.

;; Check that this produces the expected assembly output
; RUN: llc -mtriple=thumbv7-windows -o - %s -verify-machineinstrs | FileCheck %s
;; Also try to write an object file, which verifies that the SEH opcodes
;; match the actual prologue/epilogue length.
; RUN: llc -mtriple=thumbv7-windows -filetype=obj -o %t.obj %s -verify-machineinstrs

; CHECK-LABEL: d:
; CHECK: .seh_proc d

; CHECK:              push.w  {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         .seh_endprologue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         pop.w   {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         b.w     b
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         .seh_endepilogue

; CHECK:              .seh_startepilogue
; CHECK-NEXT:         pop.w   {r11, lr}
; CHECK-NEXT:         .seh_save_regs_w        {r11, lr}
; CHECK-NEXT:         b.w     c
; CHECK-NEXT:         .seh_nop_w
; CHECK-NEXT:         .seh_endepilogue
; CHECK-NEXT:         .seh_endproc

@a = global i32 0, align 4

define arm_aapcs_vfpcc void @d() optsize uwtable "frame-pointer"="none" {
entry:
  %0 = load i32, ptr @a, align 4
  switch i32 %0, label %if.then1 [
    i32 10, label %if.then
    i32 0, label %if.end2
  ]

if.then:
  tail call arm_aapcs_vfpcc void @b()
  br label %return

if.then1:
  tail call arm_aapcs_vfpcc void @b()
  br label %if.end2

if.end2:
  tail call arm_aapcs_vfpcc void @c()
  br label %return

return:
  ret void
}

declare arm_aapcs_vfpcc void @b(...)

declare arm_aapcs_vfpcc void @c(...)
