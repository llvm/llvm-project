; RUN: llc -mtriple arm-apple-darwin -filetype asm -o - %s | FileCheck -check-prefix CHECK-ARMv4 %s
; RUN: llc -mtriple armv7-apple-darwin -mcpu=cortex-a8 -filetype asm -o - %s | FileCheck -check-prefix CHECK-ARMv7 %s

define i32 @f7(float %a, float %b) {
entry:
  %tmp = fcmp ueq float %a,%b
  %retval = select i1 %tmp, i32 666, i32 42
  ret i32 %retval
}

; CHECK-ARMv4-LABEL: f7:
; CHECK-ARMv4-DAG: bl ___eqsf2
; CHECK-ARMv4-DAG: bl ___unordsf2
; CHECK-ARMv4: cmp r0, #0
; CHECK-ARMv4: movne r0, #1
; CHECK-ARMv4: orrs r0, r0,
; CHECK-ARMv4: moveq r0, #42

; CHECK-ARMv7-LABEL: f7:
; CHECK-ARMv7: vcmpe.f32
; CHECK-ARMv7: vmrs APSR_nzcv, fpscr
; CHECK-ARMv7: movweq
; CHECK-ARMv7-NOT: vmrs
; CHECK-ARMv7: movwvs

