; RUN: llc -mtriple aarch64-linux-pauthtest -o - %s \
; RUN:     -aarch64-authenticated-lr-check-method=xpac-hint \
; RUN:     -stop-before=aarch64-ptrauth \
; RUN:     | FileCheck --check-prefix=MIR %s

; RUN: llc -mtriple aarch64-linux-pauthtest -o - %s \
; RUN:     -aarch64-authenticated-lr-check-method=xpac-hint \
; RUN:     | FileCheck --check-prefix=ASM %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

; Test that expansion of AUTH_TCRETURN does not crash due to unavailability of
; neither x16 nor x17 as a scratch register.
define i32 @test_scratch_reg_nobti(ptr %callee, ptr %addr) #0 {
entry:
  ; Force spilling of LR
  tail call void asm sideeffect "", "~{lr}"()
  ; Clobber x0-x15 and x18-x29
  tail call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15}"()
  tail call void asm sideeffect "", "~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp}"()
  %addr.i = ptrtoint ptr %addr to i64
  %call = tail call i32 %callee() #1 [ "ptrauth"(i32 0, i64 %addr.i) ]
  ret i32 %call
}
; MIR-LABEL: name: test_scratch_reg_nobti
; MIR:         AUTH_TCRETURN{{ }}
;
; ASM-LABEL: @test_scratch_reg_nobti
; ASM:         autibsp
; ASM-NEXT:    eor  x17, x30, x30, lsl #1
; ASM-NEXT:    tbz  x17, #62, .Lauth_success_0
; ASM-NEXT:    brk  #0xc471
; ASM-NEXT:  .Lauth_success_0:
; ASM-NEXT:    braa x0, x16

; The same for AUTH_TCRETURN_BTI.
define i32 @test_scratch_reg_bti(ptr %callee, ptr %addr) "branch-target-enforcement" #0 {
entry:
  ; Force spilling of LR
  tail call void asm sideeffect "", "~{lr}"()
  ; Clobber x0-x15 and x18-x29
  tail call void asm sideeffect "", "~{x0},~{x1},~{x2},~{x3},~{x4},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15}"()
  tail call void asm sideeffect "", "~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{fp}"()
  %addr.i = ptrtoint ptr %addr to i64
  %call = tail call i32 %callee() #1 [ "ptrauth"(i32 0, i64 %addr.i) ]
  ret i32 %call
}
; MIR-LABEL: name: test_scratch_reg_bti
; MIR:         AUTH_TCRETURN_BTI
;
; ASM-LABEL: @test_scratch_reg_bti
; ASM:         autibsp
; ASM-NEXT:    eor  x17, x30, x30, lsl #1
; ASM-NEXT:    tbz  x17, #62, .Lauth_success_1
; ASM-NEXT:    brk  #0xc471
; ASM-NEXT:  .Lauth_success_1:
; ASM-NEXT:    braa x16, x0

attributes #0 = { nounwind "ptrauth-auth-traps" "ptrauth-calls" "ptrauth-returns" "target-features"="+pauth" }
