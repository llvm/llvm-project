; RUN: llc -mtriple=sparc < %s | FileCheck %s
; RUN: llc -mtriple=sparc -filetype=obj < %s -o %t
; RUN: llvm-objdump -r %t | FileCheck %s --check-prefix=RELOC

@fsr_storage = global i32 0, align 4

; CHECK-LABEL: get_fsr:
; CHECK:       sethi %hi(fsr_storage), %[[HI:[gilo][0-7]]]
; CHECK:       st %fsr, [%[[HI]]+%lo(fsr_storage)]
; CHECK-NEXT:  ld [%[[HI]]+%lo(fsr_storage)], %{{[gilo][0-7]}}

; RELOC:      R_SPARC_HI22 fsr_storage
; RELOC-NEXT: R_SPARC_LO10 fsr_storage
; RELOC-NOT:  R_SPARC_13 fsr_storage
define i32 @get_fsr() {
entry:
  %fsr = call i32 asm sideeffect "st\09%fsr, $1\0A\09ld\09$1, $0", "=r,*m"(ptr elementtype(i32) @fsr_storage)
  ret i32 %fsr
}
