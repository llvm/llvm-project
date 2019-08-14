; RUN: llc -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=32BIT %s

; RUN: llc -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck --check-prefix=64BIT %s

define void @bar() {
entry:

; 32BIT: mflr 0
; 32BIT: stw 0, 8(1)
; 32BIT: stwu 1, -64(1)
; 32BIT: bl .foo
; 32BIT: nop
; 32BIT: addi 1, 1, 64
; 32BIT: lwz 0, 8(1)
; 32BIT: mtlr 0

; 64BIT: mflr 0
; 64BIT: std 0, 16(1)
; 64BIT: stdu 1, -112(1)
; 64BIT: bl .foo
; 64BIT: nop
; 64BIT: addi 1, 1, 112
; 64BIT: ld 0, 16(1)
; 64BIT: mtlr 0

  call void bitcast (void (...)* @foo to void ()*)()
  ret void
}

declare void @foo(...)
