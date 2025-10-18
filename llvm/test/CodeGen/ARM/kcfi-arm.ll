; RUN: llc -mtriple=armv7-linux-gnueabi -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=armv7-linux-gnueabi -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=MIR,ISEL
; RUN: llc -mtriple=armv7-linux-gnueabi -verify-machineinstrs -stop-after=kcfi < %s | FileCheck %s --check-prefixes=MIR,KCFI

; ASM:       .long 12345678
define void @f1(ptr noundef %x) !kcfi_type !1 {
; ASM-LABEL: f1:
; ASM:       @ %bb.0:
; ASM:         bic r12, r0, #1
; ASM-NEXT:    ldr r12, [r12, #-4]
; ASM-NEXT:    eor r12, r12, #78
; ASM-NEXT:    eor r12, r12, #24832
; ASM-NEXT:    eor r12, r12, #12320768
; ASM-NEXT:    eors r12, r12, #0
; ASM-NEXT:    beq .Ltmp{{[0-9]+}}
; UDF encoding: 0x8000 | (0x1F << 5) | r0 = 0x83e0 = 33760
; ASM-NEXT:    udf #33760
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    blx r0

; MIR-LABEL: name: f1
; MIR: body:

; ISEL:     BLX %0, csr_aapcs,{{.*}} cfi-type 12345678

; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r0, 12345678
; KCFI-NEXT:    BLX killed $r0, csr_aapcs,{{.*}}
; KCFI-NEXT:  }

  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; Test with tail call
define void @f2(ptr noundef %x) !kcfi_type !1 {
; ASM-LABEL: f2:
; ASM:       @ %bb.0:
; ASM:         bic r12, r0, #1
; ASM:         ldr r12, [r12, #-4]
; ASM:         eor r12, r12, #78
; ASM:         eor r12, r12, #24832
; ASM:         eor r12, r12, #12320768
; ASM:         eors r12, r12, #0
; ASM:         beq .Ltmp{{[0-9]+}}
; UDF encoding: 0x8000 | (0x1F << 5) | r0 = 0x83e0 = 33760
; ASM:         udf #33760
; ASM:       .Ltmp{{[0-9]+}}:
; ASM:         bx r0

; MIR-LABEL: name: f2
; MIR: body:

; ISEL:     TCRETURNri %0, 0, csr_aapcs, implicit $sp, cfi-type 12345678

; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $r0, 12345678
; KCFI-NEXT:    TAILJMPr killed $r0, csr_aapcs, implicit $sp, implicit $sp
; KCFI-NEXT:  }

  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; Test r3 spill/reload when target is r12 and r3 is a call argument.
; With 5+ arguments (target + 4 args), r0-r3 are all used for arguments,
; forcing r3 to be spilled when we need it as scratch register.
define void @f3_r3_spill(ptr noundef %target, i32 %a, i32 %b, i32 %c, i32 %d) !kcfi_type !1 {
; ASM-LABEL: f3_r3_spill:
; ASM:       @ %bb.0:
; Arguments: r0=%target, r1=%a, r2=%b, r3=%c, [sp]=%d
; Call needs: r0=%a, r1=%b, r2=%c, r3=%d, target in r12
; Compiler shuffles arguments into place, saving r3 (c) in lr, loading d from stack
; ASM:         mov lr, r3
; ASM-NEXT:    ldr r3, [sp, #8]
; ASM-NEXT:    mov r12, r0
; ASM-NEXT:    mov r0, r1
; ASM-NEXT:    mov r1, r2
; ASM-NEXT:    mov r2, lr
; r3 is live as 4th argument, so push it before KCFI check
; ASM-NEXT:    stmdb sp!, {r3}
; ASM-NEXT:    bic r3, r12, #1
; ASM-NEXT:    ldr r3, [r3, #-4]
; ASM-NEXT:    eor r3, r3, #78
; ASM-NEXT:    eor r3, r3, #24832
; ASM-NEXT:    eor r3, r3, #12320768
; ASM-NEXT:    eors r3, r3, #0
; Restore r3 immediately after comparison, before branch
; ASM-NEXT:    ldm sp!, {r3}
; ASM-NEXT:    beq .Ltmp{{[0-9]+}}
; UDF encoding: 0x8000 | (0x1F << 5) | r12 = 0x83ec = 33772
; ASM-NEXT:    udf #33772
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    blx r12
;
  call void %target(i32 %a, i32 %b, i32 %c, i32 %d) [ "kcfi"(i32 12345678) ]
  ret void
}

; Test with 3 arguments - r3 not live, target in r12, so r3 used as scratch without spilling
define void @f4_r3_unused(ptr noundef %target, i32 %a, i32 %b) !kcfi_type !1 {
; ASM-LABEL: f4_r3_unused:
; ASM:       @ %bb.0:
; Only 3 arguments total, so r3 is not used as call argument
; Compiler puts target→r3, a→r0, b→r1
; ASM:         mov r3, r0
; ASM-NEXT:    mov r0, r1
; ASM-NEXT:    mov r1, r2
; r3 is the target, so we use r12 as scratch (no spill needed)
; ASM-NEXT:    bic r12, r3, #1
; ASM-NEXT:    ldr r12, [r12, #-4]
; ASM-NEXT:    eor r12, r12, #78
; ASM-NEXT:    eor r12, r12, #24832
; ASM-NEXT:    eor r12, r12, #12320768
; ASM-NEXT:    eors r12, r12, #0
; ASM-NEXT:    beq .Ltmp{{[0-9]+}}
; UDF encoding: 0x8000 | (0x1F << 5) | r3 = 0x83e3 = 33763
; ASM-NEXT:    udf #33763
; ASM-NEXT:  .Ltmp{{[0-9]+}}:
; ASM-NEXT:    blx r3
;
  call void %target(i32 %a, i32 %b) [ "kcfi"(i32 12345678) ]
  ret void
}

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
