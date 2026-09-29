; RUN: llc -mtriple=thumbv7-windows-itanium -mcpu=cortex-a9 -o - %s \
; RUN:  | FileCheck %s -check-prefix CHECK-SMALL-CODE
; RUN: llc -mtriple=thumbv7-windows-itanium -mcpu=cortex-a9 -code-model=large -o - %s \
; RUN:  | FileCheck %s -check-prefix CHECK-LARGE-CODE
; RUN: llc -mtriple=thumbv7-windows-msvc -mcpu=cortex-a9 -o - %s \
; RUN:  | FileCheck %s -check-prefix CHECK-SMALL-CODE

define arm_aapcs_vfpcc i8 @function(i32 %sz, i32 %idx) {
entry:
  %vla = alloca i8, i32 %sz, align 1
  %arrayidx = getelementptr inbounds i8, ptr %vla, i32 %idx
  %0 = load volatile i8, ptr %arrayidx, align 1
  ret i8 %0
}

; CHECK-SMALL-CODE:   adds [[R4:r[0-9]+]], #7
; CHECK-SMALL-CODE:   bic [[R4]], [[R4]], #4
; CHECK-SMALL-CODE:   lsrs r4, [[R4]], #2
; CHECK-SMALL-CODE:   bl __chkstk
; CHECK-SMALL-CODE:   sub.w sp, sp, r4

; CHECK-LARGE-CODE:   adds  [[R4:r[0-9]+]], #7
; CHECK-LARGE-CODE:   bic   [[R4]], [[R4]], #4
; CHECK-LARGE-CODE:   lsrs  r4, [[R4]], #2
; CHECK-LARGE-CODE:   movw  [[IP:r[0-9]+]], :lower16:__chkstk
; CHECK-LARGE-CODE:   movt  [[IP]], :upper16:__chkstk
; CHECK-LARGE-CODE:   blx   [[IP]]
; CHECK-LARGE-CODE:   sub.w sp, sp, r4
