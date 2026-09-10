; RUN: llc -mtriple=sparc < %s | FileCheck %s

@fsr_storage = global i32 0, align 4

; CHECK-LABEL: get_fsr:
; CHECK:       sethi %hi(fsr_storage), %[[HI:[gilo][0-7]]]
; CHECK:       st %fsr, [%[[HI]]+%lo(fsr_storage)]
; CHECK-NEXT:  ld [%[[HI]]+%lo(fsr_storage)], %{{[gilo][0-7]}}

define i32 @get_fsr() {
entry:
  %fsr = call i32 asm sideeffect "st\09%fsr, $1\0A\09ld\09$1, $0", "=r,*m"(ptr elementtype(i32) @fsr_storage)
  ret i32 %fsr
}
