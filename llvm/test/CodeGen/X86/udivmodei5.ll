; RUN: llc < %s -mtriple=i686-unknown-unknown | FileCheck %s --check-prefix=X86
; RUN: llc < %s -mtriple=x86_64-unknown-unknown | FileCheck %s --check-prefix=X64

; On i686, this is expanded into a loop. On x86_64, this calls __udivti3.
define i65 @udiv65(i65 %a, i65 %b) nounwind {
; X86-LABEL: udiv65:
; X86-NOT:     call
;
; X64-LABEL: udiv65:
; X64:       # %bb.0:
; X64-NEXT:    pushq %rax
; X64-NEXT:    andl $1, %esi
; X64-NEXT:    andl $1, %ecx
; X64-NEXT:    callq __udivti3@PLT
; X64-NEXT:    popq %rcx
; X64-NEXT:    retq
  %res = udiv i65 %a, %b
  ret i65 %res
}

define i129 @udiv129(i129 %a, i129 %b) nounwind {
; X86-LABEL: udiv129:
; X86-NOT:     call
;
; X64-LABEL: udiv129:
; X64-NOT:     call
  %res = udiv i129 %a, %b
  ret i129 %res
}

define i129 @urem129(i129 %a, i129 %b) nounwind {
; X86-LABEL: urem129:
; X86-NOT:     call
;
; X64-LABEL: urem129:
; X64-NOT:     call
  %res = urem i129 %a, %b
  ret i129 %res
}

define i129 @sdiv129(i129 %a, i129 %b) nounwind {
; X86-LABEL: sdiv129:
; X86-NOT:     call
;
; X64-LABEL: sdiv129:
; X64-NOT:     call
  %res = sdiv i129 %a, %b
  ret i129 %res
}

define i129 @srem129(i129 %a, i129 %b) nounwind {
; X86-LABEL: srem129:
; X86-NOT:     call
;
; X64-LABEL: srem129:
; X64-NOT:     call
  %res = srem i129 %a, %b
  ret i129 %res
}

; Some higher sizes
define i257 @sdiv257(i257 %a, i257 %b) nounwind {
; X86-LABEL: sdiv257:
; X86-NOT:     call
;
; X64-LABEL: sdiv257:
; X64-NOT:     call
  %res = sdiv i257 %a, %b
  ret i257 %res
}
