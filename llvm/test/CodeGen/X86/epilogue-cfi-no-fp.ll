; RUN: llc -O0 < %s | FileCheck %s
target triple = "i686--"

; Function Attrs: noinline nounwind
define i32 @foo(i32 %i, i32 %j, i32 %k, i32 %l, i32 %m) {
; CHECK-LABEL:   foo:
; CHECK:         popl   %esi
; CHECK-NEXT:    .cfi_def_cfa_offset 16
; CHECK-NEXT:    popl	%edi
; CHECK-NEXT:    .cfi_def_cfa_offset 12
; CHECK-NEXT:    popl	%ebx
; CHECK-NEXT:    .cfi_def_cfa_offset 8
; CHECK-NEXT:    popl	%ebp
; CHECK-NEXT:    .cfi_def_cfa_offset 4
; CHECK-NEXT:    retl
entry:
  tail call void asm sideeffect "nop", "~{eax},~{ebx},~{ecx},~{edx},~{esi},~{edi},~{ebp}"()
  ret i32 0
}
