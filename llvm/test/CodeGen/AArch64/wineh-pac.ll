; RUN: llc < %s -mtriple=aarch64-windows | FileCheck %s

define dso_local i32 @func(ptr %g, i32 %a) {
entry:
  tail call void %g() #2
  ret i32 %a
}

define dso_local i32 @func2(ptr %g, i32 %a) "target-features"="+v8.3a" {
entry:
  tail call void %g() #2
  ret i32 %a
}

!llvm.module.flags = !{!0}

!0 = !{i32 8, !"sign-return-address", i32 1}

; CHECK-LABEL: func:
; CHECK-NEXT: .seh_proc func
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: hint #27
; CHECK-NEXT: .seh_pac_sign_lr
; CHECK-NEXT: str x19, [sp, #-16]!
; CHECK-NEXT: .seh_save_reg_x x19, 16
; CHECK-NEXT: str x30, [sp, #8]
; CHECK-NEXT: .seh_save_reg x30, 8
; CHECK-NEXT: .seh_endprologue

; CHECK:      .seh_startepilogue
; CHECK-NEXT: ldr x30, [sp, #8]
; CHECK-NEXT: .seh_save_reg x30, 8
; CHECK-NEXT: ldr x19, [sp], #16
; CHECK-NEXT: .seh_save_reg_x x19, 16
; CHECK-NEXT: hint #31
; CHECK-NEXT: .seh_pac_sign_lr
; CHECK-NEXT: .seh_endepilogue
; CHECK-NEXT: ret
; CHECK-NEXT: .seh_endfunclet
; CHECK-NEXT: .seh_endproc

;; For func2, check that the potentially folded autibsp+ret -> retab
;; is handled correctly - currently we inhibit producing retab here.

; CHECK-LABEL: func2:
; CHECK-NEXT: .seh_proc func2
; CHECK-NEXT: // %bb.0:
; CHECK-NEXT: pacibsp
; CHECK-NEXT: .seh_pac_sign_lr
; CHECK-NEXT: str x19, [sp, #-16]!
; CHECK-NEXT: .seh_save_reg_x x19, 16
; CHECK-NEXT: str x30, [sp, #8]
; CHECK-NEXT: .seh_save_reg x30, 8
; CHECK-NEXT: .seh_endprologue

; CHECK:      .seh_startepilogue
; CHECK-NEXT: ldr x30, [sp, #8]
; CHECK-NEXT: .seh_save_reg x30, 8
; CHECK-NEXT: ldr x19, [sp], #16
; CHECK-NEXT: .seh_save_reg_x x19, 16
; CHECK-NEXT: autibsp
; CHECK-NEXT: .seh_pac_sign_lr
; CHECK-NEXT: .seh_endepilogue
; CHECK-NEXT: ret
; CHECK-NEXT: .seh_endfunclet
; CHECK-NEXT: .seh_endproc
