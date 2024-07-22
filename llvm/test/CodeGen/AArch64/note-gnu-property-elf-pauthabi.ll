; RUN: rm -rf %t && split-file %s %t && cd %t

;--- ok.ll

; RUN: llc -mtriple=aarch64-linux ok.ll               -o - | \
; RUN:   FileCheck %s --check-prefix=ASM
; RUN: llc -mtriple=aarch64-linux ok.ll -filetype=obj -o - |  \
; RUN:   llvm-readelf --notes - | FileCheck %s --check-prefix=OBJ

!llvm.module.flags = !{!0, !1}

!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 268435458}
!1 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 85}

; ASM: .section .note.gnu.property,"a",@note
; ASM-NEXT: .p2align 3, 0x0
; ASM-NEXT: .word 4
; ASM-NEXT: .word 24
; ASM-NEXT: .word 5
; ASM-NEXT: .asciz "GNU"
; 3221225473 = 0xc0000001 = GNU_PROPERTY_AARCH64_FEATURE_PAUTH
; ASM-NEXT: .word 3221225473
; ASM-NEXT: .word 16
; ASM-NEXT: .xword 268435458
; ASM-NEXT: .xword 85

; OBJ: Displaying notes found in: .note.gnu.property
; OBJ-NEXT:   Owner                 Data size	Description
; OBJ-NEXT:   GNU                   0x00000018	NT_GNU_PROPERTY_TYPE_0 (property note)
; OBJ-NEXT:   AArch64 PAuth ABI core info: platform 0x10000002 (llvm_linux), version 0x55 (PointerAuthIntrinsics, !PointerAuthCalls, PointerAuthReturns, !PointerAuthAuthTraps, PointerAuthVTPtrAddressDiscrimination, !PointerAuthVTPtrTypeDiscrimination, PointerAuthInitFini)

; ERR: either both or no 'aarch64-elf-pauthabi-platform' and 'aarch64-elf-pauthabi-version' module flags must be present

;--- err1.ll

; RUN: not llc -mtriple=aarch64-linux err1.ll 2>&1 -o - | \
; RUN:   FileCheck %s --check-prefix=ERR

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"aarch64-elf-pauthabi-platform", i32 2}

;--- err2.ll

; RUN: not llc -mtriple=aarch64-linux err2.ll 2>&1 -o - | \
; RUN:   FileCheck %s --check-prefix=ERR

!llvm.module.flags = !{!0}

!0 = !{i32 1, !"aarch64-elf-pauthabi-version", i32 31}
