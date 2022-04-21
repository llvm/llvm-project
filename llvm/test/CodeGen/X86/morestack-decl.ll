; RUN: llc -mcpu=generic -mtriple=x86_64-linux -code-model=large < %s | FileCheck %s

; Check what happens if we have an existing declaration of __morestack_addr

; CHECK:	.section	".note.GNU-stack","",@progbits
; CHECK-NEXT:	.section	.rodata,"a",@progbits
; CHECK-NEXT: __morestack_addr:
; CHECK-NEXT: .quad	__morestack

declare void @__morestack_addr()
