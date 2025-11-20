; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx512f -show-mc-encoding | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx512f,+egpr -show-mc-encoding | FileCheck --check-prefix=EGPR %s

define void @kmov(i1 %cmp23.not) {
; CHECK-LABEL: kmov:
; CHECK:       kmovw %edi, %k1 # encoding: [0xc5,0xf8,0x92,0xcf]
;
; EGPR-LABEL: kmov:
; EGPR:       kmovw %edi, %k1 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x92,0xcf]
entry:
  %0 = select i1 %cmp23.not, double 1.000000e+00, double 0.000000e+00
  store double %0, ptr null, align 8
  ret void
}
