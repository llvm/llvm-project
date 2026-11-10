; RUN: llc -mtriple=aarch64 -O2 -enable-subreg-liveness=false < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64 -O2 -enable-subreg-liveness=true < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64 -O2 -enable-subreg-liveness=false -stop-before=register-coalescer \
; RUN:   -o - < %s | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=aarch64 -O2 -enable-subreg-liveness=true -stop-before=register-coalescer \
; RUN:   -o - < %s | FileCheck %s --check-prefix=MIR

; Ensure that the final assembly has a mov w instruction to zero the upper x bits.

; The volatile load keeps the 64-bit load even though only its low 32 bits
; are used.
define i64 @zext_loaded_i64(ptr %p) {
; ASM-LABEL: zext_loaded_i64:
; ASM:       ldr x[[REG:[0-9]+]], [x0]
; ASM-NEXT:  mov w0, w[[REG]]
; ASM-NEXT:  ret
;
; MIR-LABEL: name: zext_loaded_i64
; MIR:       %[[LOAD:[0-9]+]]:gpr64 = LDRXui
; MIR-NEXT:  %[[LOW:[0-9]+]]:gpr32 = COPY %[[LOAD]].sub_32
; MIR-NEXT:  %[[ZERO:[0-9]+]]:gpr32 = ORRWrr $wzr, %[[LOW]]
; MIR-NEXT:  %[[WIDE:[0-9]+]]:gpr64all = SUBREG_TO_REG %[[ZERO]], %subreg.sub_32
; MIR-NEXT:  $x0 = COPY %[[WIDE]]
  %wide = load volatile i64, ptr %p, align 8
  %low = trunc i64 %wide to i32
  %result = zext i32 %low to i64
  ret i64 %result
}

define i64 @zext_i32_arg(i32 %v) {
; ASM-LABEL: zext_i32_arg:
; ASM:  mov w0, w0
; ASM-NEXT:  ret
;
; MIR-LABEL: name: zext_i32_arg
; MIR:  %[[COPY:[0-9]+]]:gpr32 = COPY $w0
; MIR-NEXT:  %[[ZERO:[0-9]+]]:gpr32 = ORRWrs $wzr, %[[COPY]], 0
; MIR-NEXT:  %[[WIDE:[0-9]+]]:gpr64all = SUBREG_TO_REG %[[ZERO]], %subreg.sub_32
; MIR-NEXT:  $x0 = COPY %[[WIDE]]
  %result = zext i32 %v to i64
  ret i64 %result
}
