; RUN: llc -mtriple x86_64-unknown-linux-gnu %s -o %t.o -filetype=obj
; RUN: llvm-readobj -n %t.o | FileCheck %s

module asm ".pushsection \22.note.gnu.property\22,\22a\22,@note"
module asm "     .p2align 3"
module asm "     .long 1f - 0f"
module asm "     .long 4f - 1f"
module asm "     .long 5"
module asm "0:   .asciz \22GNU\22"
module asm "1:   .p2align 3"
module asm "     .long 0xc0008002"
module asm "     .long 3f - 2f"
module asm "2:   .long ((1U << 0) | 0 | 0 | 0)"
module asm "3:   .p2align 3"
module asm "4:"
module asm " .popsection"

!llvm.module.flags = !{!0, !1}

!0 = !{i32 4, !"cf-protection-return", i32 1}
!1 = !{i32 4, !"cf-protection-branch", i32 1}

; CHECK:      Type: NT_GNU_PROPERTY_TYPE_0
; CHECK-NEXT: Property [
; CHECK-NEXT:   x86 feature: IBT, SHSTK
; CHECK-NEXT: ]
; CHECK:      Type: NT_GNU_PROPERTY_TYPE_0
; CHECK-NEXT: Property [
; CHECK-NEXT:   x86 ISA needed: x86-64-baseline
; CHECK-NEXT: ]
