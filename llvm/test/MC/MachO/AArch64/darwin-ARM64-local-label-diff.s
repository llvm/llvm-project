; RUN: llvm-mc -triple arm64-apple-darwin -filetype=obj -o - < %s | llvm-readobj -S --sd | FileCheck %s
; rdar://13028719

 .globl context_save0
 .align 6
Lcontext_save0:
context_save0:
 .fill 2, 8, 5
Lcontext_save0_end:
Lcontext_save0_size: .quad (Lcontext_save0_end - Lcontext_save0)

 .align 6
Lcontext_save1:
 .fill 2, 8, 0
Lcontext_save1_end:
Lcontext_save1_size: .quad (Lcontext_save1_end - Lcontext_save1)

Llockup_release:
 .quad 0

; CHECK: SectionData (
; CHECK:   0000: 05000000 00000000 05000000 00000000  |................|
; CHECK:   0010: 10000000 00000000 1F2003D5 1F2003D5  |......... ... ..|
; CHECK:   0020: 1F2003D5 1F2003D5 1F2003D5 1F2003D5  |. ... ... ... ..|
; CHECK:   0030: 1F2003D5 1F2003D5 1F2003D5 1F2003D5  |. ... ... ... ..|
; CHECK:   0040: 00000000 00000000 00000000 00000000  |................|
; CHECK:   0050: 10000000 00000000 00000000 00000000  |................|
; CHECK: )
