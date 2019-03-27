; RUN: llc -mcpu=generic -mtriple=powerpc64le-unknown-unknown -O0 < %s \ 
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=GENERIC
; RUN: llc -mcpu=ppc -mtriple=powerpc64le-unknown-unknown -O0 < %s \
; RUN:   -verify-machineinstrs | FileCheck %s

define float @testRSP(double %x) {
entry:
  %0 = fptrunc double %x to float
  ret float %0
; CHECK: frsp 1, 1
; GENERIC: xsrsp 1, 1
}

