; RUN: llc --mtriple=loongarch64 --verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=ASM
; RUN: llc --mtriple=loongarch64 --print-after-isel -o /dev/null 2>&1 < %s \
; RUN:   | FileCheck %s --check-prefix=MACHINE-INSTR

define i64 @k_variable_offset(ptr %p, i64 %idx) nounwind {
; ASM-LABEL: k_variable_offset:
; ASM:       # %bb.0:
; ASM-NEXT:    #APP
; ASM-NEXT:    ldx.d $a0, $a0, $a1
; ASM-NEXT:    #NO_APP
; ASM-NEXT:    ret
  %1 = getelementptr inbounds i8, ptr %p, i64 %idx
;; Make sure machine instr with this 'k' constraint is printed correctly.
; MACHINE-INSTR: INLINEASM{{.*}}[mem:k]
  %2 = call i64 asm "ldx.d $0, $1", "=r,*k"(ptr elementtype(i64) %1)
  ret i64 %2
}

define i64 @k_constant_offset(ptr %p) nounwind {
; ASM-LABEL: k_constant_offset:
; ASM:       # %bb.0:
; ASM-NEXT:    ori $a1, $zero, 5
; ASM-NEXT:    #APP
; ASM-NEXT:    ldx.d $a0, $a0, $a1
; ASM-NEXT:    #NO_APP
; ASM-NEXT:    ret
  %1 = getelementptr inbounds i8, ptr %p, i64 5
;; Make sure machine instr with this 'k' constraint is printed correctly.
; MACHINE-INSTR: INLINEASM{{.*}}[mem:k]
  %2 = call i64 asm "ldx.d $0, $1", "=r,*k"(ptr elementtype(i64) %1)
  ret i64 %2
}
