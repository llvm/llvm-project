; RUN: llc --mtriple=loongarch64 --verify-machineinstrs < %s \
; RUN:   | FileCheck %s --check-prefix=ASM
; RUN: llc --mtriple=loongarch64 --print-after-isel -o /dev/null 2>&1 < %s \
; RUN:   | FileCheck %s --check-prefix=MACHINE-INSTR

;; Note amswap.w is not available on loongarch32.

define void @ZB(ptr %p) nounwind {
; ASM-LABEL: ZB:
; ASM:       # %bb.0:
; ASM-NEXT:    #APP
; ASM-NEXT:    amswap.w $t0, $t1, $a0
; ASM-NEXT:    #NO_APP
; ASM-NEXT:    ret
;; Make sure machine instr with this "ZB" constraint is printed correctly.
; MACHINE-INSTR: INLINEASM{{.*}}[mem:ZB]
  call void asm "amswap.w $$r12, $$r13, $0", "*^ZB"(ptr elementtype(i32) %p)
  ret void
}

define void @ZB_constant_offset(ptr %p) nounwind {
; ASM-LABEL: ZB_constant_offset:
; ASM:       # %bb.0:
; ASM-NEXT:    addi.d $a0, $a0, 1
; ASM-NEXT:    #APP
; ASM-NEXT:    amswap.w $t0, $t1, $a0
; ASM-NEXT:    #NO_APP
; ASM-NEXT:    ret
  %1 = getelementptr inbounds i8, ptr %p, i32 1
;; Make sure machine instr with this "ZB" constraint is printed correctly.
; MACHINE-INSTR: INLINEASM{{.*}}[mem:ZB]
  call void asm "amswap.w $$r12, $$r13, $0", "*^ZB"(ptr elementtype(i32) %1)
  ret void
}

define void @ZB_variable_offset(ptr %p, i32 signext %idx) nounwind {
; ASM-LABEL: ZB_variable_offset:
; ASM:       # %bb.0:
; ASM-NEXT:    add.d $a0, $a0, $a1
; ASM-NEXT:    #APP
; ASM-NEXT:    amswap.w $t0, $t1, $a0
; ASM-NEXT:    #NO_APP
; ASM-NEXT:    ret
  %1 = getelementptr inbounds i8, ptr %p, i32 %idx
;; Make sure machine instr with this "ZB" constraint is printed correctly.
; MACHINE-INSTR: INLINEASM{{.*}}[mem:ZB]
  call void asm "amswap.w $$r12, $$r13, $0", "*^ZB"(ptr elementtype(i32) %1)
  ret void
}
