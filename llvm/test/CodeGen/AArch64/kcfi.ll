; RUN: llc -mtriple=aarch64-- -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -global-isel < %s | FileCheck %s --check-prefix=ASM

; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=MIR,ISEL
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -stop-after=finalize-isel -global-isel < %s | FileCheck %s --check-prefixes=MIR,ISEL

; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -mattr=harden-sls-blr -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=MIR,ISEL-SLS
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -mattr=harden-sls-blr -stop-after=finalize-isel -global-isel < %s | FileCheck %s --check-prefixes=MIR,ISEL-SLS

; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -stop-after=aarch64-kcfi < %s | FileCheck %s --check-prefixes=MIR,KCFI
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -mattr=harden-sls-blr -stop-after=aarch64-kcfi < %s | FileCheck %s --check-prefixes=MIR,KCFI-SLS

; ASM:       .word 12345678
define void @f1(ptr noundef %x) !kcfi_type !1 {
; ASM-LABEL: f1:
; ASM:       // %bb.0:
; ASM:         ldur w16, [x0, #-4]
; ASM-NEXT:    movk w17, #24910
; ASM-NEXT:    movk w17, #188, lsl #16
; ASM-NEXT:    cmp w16, w17
; ASM-NEXT:    b.eq .Ltmp0
; ASM-NEXT:    brk #0x8220
; ASM-NEXT:  .Ltmp0:
; ASM-NEXT:    blr x0

; MIR-LABEL: name: f1
; MIR: body:

; ISEL:     BLR %0, csr_aarch64_aapcs,{{.*}} cfi-type 12345678
; ISEL-SLS: BLRNoIP %0, csr_aarch64_aapcs,{{.*}} cfi-type 12345678

; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; KCFI-NEXT:    BLR killed $x0, csr_aarch64_aapcs,{{.*}}
; KCFI-NEXT:  }

; KCFI-SLS:       BUNDLE{{.*}} {
; KCFI-SLS-NEXT:    KCFI_CHECK $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; KCFI-SLS-NEXT:    BLRNoIP killed $x0, csr_aarch64_aapcs,{{.*}}
; KCFI-SLS-NEXT:  }

  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; ASM-NOT: .word:
define void @f2(ptr noundef %x) #0 {
; ASM-LABEL: f2:
; ASM:       // %bb.0:
; ASM-NEXT:    nop
; ASM-NEXT:    nop
; ASM:         ldur w16, [x0, #-4]
; ASM-NEXT:    movk w17, #24910
; ASM-NEXT:    movk w17, #188, lsl #16
; ASM-NEXT:    cmp w16, w17
; ASM-NEXT:    b.eq .Ltmp1
; ASM-NEXT:    brk #0x8220
; ASM-NEXT:  .Ltmp1:
; ASM-NEXT:    br x0

; MIR-LABEL: name: f2
; MIR: body:

; ISEL:     TCRETURNri %0, 0, csr_aarch64_aapcs, implicit $sp, cfi-type 12345678

; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; KCFI-NEXT:    TCRETURNri killed $x0, 0, csr_aarch64_aapcs, implicit $sp
; KCFI-NEXT:  }

  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { "patchable-function-entry"="2" }

!llvm.module.flags = !{!0}
!0 = !{i32 4, !"kcfi", i32 1}
!1 = !{i32 12345678}
