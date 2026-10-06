; RUN: llvm-ml -m64 -filetype=s %s /Fo - | FileCheck %s

; Before any section: there is no section to switch back to.
includelib first_lib

; CHECK:      .section .drectve,"yni"
; CHECK-NEXT: .ascii "/DEFAULTLIB:"
; CHECK-NEXT: .ascii "first_lib"
; CHECK-NEXT: .byte 32
; CHECK-NOT:  .section
; CHECK:      .text

.code

; Inside a section: switch back to it.
includelib second_lib

; CHECK:      .section .drectve,"yni"
; CHECK-NEXT: .ascii "/DEFAULTLIB:"
; CHECK-NEXT: .ascii "second_lib"
; CHECK-NEXT: .byte 32
; CHECK-NEXT: .text

end
