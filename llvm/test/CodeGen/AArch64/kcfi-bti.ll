; RUN: llc -mtriple=aarch64-- -verify-machineinstrs < %s | FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -stop-after=finalize-isel < %s | FileCheck %s --check-prefixes=MIR,ISEL
; RUN: llc -mtriple=aarch64-- -verify-machineinstrs -stop-after=kcfi < %s | FileCheck %s --check-prefixes=MIR,KCFI

; ASM:       .word 12345678
define void @f1(ptr noundef %x) #1 !kcfi_type !2 {
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

; ISEL: BLR %0, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp, cfi-type 12345678

; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; KCFI-NEXT:    BLR killed $x0, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp
; KCFI-NEXT:  }

  call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

; ASM:       .word 12345678
define void @f2(ptr noundef %x)  #1 !kcfi_type !2 {
; ASM-LABEL: f2:
; ASM:       // %bb.0:
; ASM:         ldur w16, [x0, #-4]
; ASM-NEXT:    movk w17, #24910
; ASM-NEXT:    movk w17, #188, lsl #16
; ASM-NEXT:    cmp w16, w17
; ASM-NEXT:    b.eq .Ltmp1
; ASM-NEXT:    brk #0x8220
; ASM-NEXT:  .Ltmp1:
; ASM-NEXT:    blr x0

; MIR-LABEL: name: f2
; MIR: body:

; ISEL: BLR_BTI %0, csr_aarch64_aapcs, implicit-def dead $lr, implicit $sp, implicit-def $sp, cfi-type 12345678

; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $x0, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; KCFI-NEXT:    BLR killed $x0, csr_aarch64_aapcs, implicit-def $lr, implicit $sp, implicit-def dead $lr, implicit $sp, implicit-def $sp
; KCFI-NEXT:    HINT 36
; KCFI-NEXT:  }

  call void %x() #0 [ "kcfi"(i32 12345678) ]
  ret void
}

; ASM-NOT: .word:
define void @f3(ptr noundef %x) #1 {
; ASM-LABEL: f3:
; ASM:       // %bb.0:
; ASM:         ldur w9, [x16, #-4]
; ASM-NEXT:    movk w17, #24910
; ASM-NEXT:    movk w17, #188, lsl #16
; ASM-NEXT:    cmp w9, w17
; ASM-NEXT:    b.eq .Ltmp2
; ASM-NEXT:    brk #0x8230
; ASM-NEXT:  .Ltmp2:
; ASM-NEXT:    br x16

; MIR-LABEL: name: f3
; MIR: body:

; ISEL: TCRETURNrix16x17 %1, 0, csr_aarch64_aapcs, implicit $sp, cfi-type 12345678

; KCFI:       BUNDLE{{.*}} {
; KCFI-NEXT:    KCFI_CHECK $x16, 12345678, implicit-def $x9, implicit-def $x16, implicit-def $x17, implicit-def $nzcv
; KCFI-NEXT:    TCRETURNrix16x17 internal killed $x16, 0, csr_aarch64_aapcs, implicit $sp
; KCFI-NEXT:  }

  tail call void %x() [ "kcfi"(i32 12345678) ]
  ret void
}

attributes #0 = { returns_twice }
attributes #1 = { "branch-target-enforcement" }

!llvm.module.flags = !{!0, !1}
!0 = !{i32 8, !"branch-target-enforcement", i32 1}
!1 = !{i32 4, !"kcfi", i32 1}
!2 = !{i32 12345678}
