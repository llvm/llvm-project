; RUN: llc  -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff -mattr=-altivec \
; RUN:      -mcpu=pwr4 --ppc-enable-load-store-multiple < %s | FileCheck %s

target triple = "powerpc-ibm-aix7.2.0.0"

define dso_local void @test_simple() #0 {
entry:
  call void asm sideeffect "nop", "~{r16},~{r17},~{r18},~{r19},~{r20},~{r21},~{r22},~{r23},~{r24},~{r25},~{r26},~{r27},~{r28},~{r29},~{r30},~{r31}"()
  ret void
}

; CHECK:        stmw 16, -64(1)                         # 4-byte Folded Spill
; CHECK-NEXT:   #APP
; CHECK-NEXT:   nop
; CHECK-NEXT:   #NO_APP
; CHECK-NEXT:   lmw 16, -64(1)                          # 4-byte Folded Reload
