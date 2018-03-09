; RUN: llc -relocation-model=pic < %s | FileCheck %s

; The load restoring r30 at the end of the function was placed out of order
; relative to its uses as the PIC base pointer.
; This was because the r30 operand was not marked as "def" which allowed
; the post-RA scheduler to move it over other uses of r30.

; CHECK-LABEL: fred
; CHECK:       lwz 30, 24(1)
; R30 should not appear in an instruction after it's been restored.
; CHECK-NOT:   30,

target datalayout = "E-m:e-p:32:32-i64:64-n32"
target triple = "powerpc--"

define double @fred(i64 %a) #0 {
entry:
  %0 = lshr i64 %a, 32
  %conv = trunc i64 %0 to i32
  %conv1 = sitofp i32 %conv to double
  %mul = fmul double %conv1, 0x41F0000000000000
  %and = and i64 %a, 4294967295
  %or = or i64 %and, 4841369599423283200
  %sub = fadd double %mul, 0xC330000000000000
  %1 = bitcast i64 %or to double
  %add = fadd double %sub, %1
  ret double %add
}

attributes #0 = { norecurse nounwind readnone "target-cpu"="ppc" "use-soft-float"="false" }
