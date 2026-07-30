; REQUIRES: asserts
; RUN: llc < %s -mtriple=x86_64-linux   | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-linux -regalloc=greedy | FileCheck %s
; RUN: llc < %s -mtriple=i386-linux -mattr=+sse2 | FileCheck %s
; CHECK:     LCPI
; CHECK:     LCPI
; CHECK:     LCPI
; CHECK-NOT: LCPI

; RUN: llc < %s -mtriple=x86_64-linux -o /dev/null -stats -info-output-file - | FileCheck %s -check-prefix=X64stat
; X64stat: 6 asm-printer

; RUN: llc < %s -mtriple=i686-- -mattr=+sse2 -o /dev/null -stats -info-output-file - | FileCheck %s -check-prefix=X32stat
; X32stat: 12 asm-printer

declare float @qux(float %y)

define float @array(float %a) nounwind {
  %n = fmul float %a, 9.0
  %m = call float @qux(float %n)
  %o = fmul float %m, 9.0
  ret float %o
}
