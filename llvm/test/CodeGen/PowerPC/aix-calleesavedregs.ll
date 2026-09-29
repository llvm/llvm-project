; RUN: llc -verify-machineinstrs -mcpu=pwr4 -mattr=-altivec \
; RUN:  -mtriple powerpc-ibm-aix-xcoff -O0 < %s | \
; RUN: FileCheck --check-prefixes=CHECK %s

define void @usethirteen() {
    call void asm "nop", "~{r13}"()
    ret void
}

; CHECK: stw 13, -76(1)
; CHECK: lwz 13, -76(1)
