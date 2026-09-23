; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -O0 -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck %s
; RUN: llc -mtriple=mipsel -mcpu=mips32r2 -O2 -debug-pass=Structure -o /dev/null %s 2>&1 | FileCheck %s

; Expand LL/SC loops after generic machine passes and before MIPS finalization.
; CHECK:      Live DEBUG_VALUE analysis
; CHECK:      Stack Frame Layout Analysis
; CHECK-NEXT: Mips pseudo instruction expansion pass
; CHECK-NEXT: microMIPS instruction size reduction pass
; CHECK-NEXT: Mips Delay Slot Filler
; CHECK-NEXT: Mips Branch Expansion Pass
; CHECK-NEXT: Mips Constant Islands

define i8 @add(ptr %ptr, i8 %val) {
  %old = atomicrmw add ptr %ptr, i8 %val monotonic
  ret i8 %old
}
