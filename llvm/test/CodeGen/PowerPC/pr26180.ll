; RUN: llc -mcpu=generic -mtriple=powerpc64le-unknown-unknown -O0 < %s \ 
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=GENERIC
; RUN: llc -mcpu=ppc -mtriple=powerpc64le-unknown-unknown -O0 < %s \
; RUN:   -verify-machineinstrs | FileCheck %s

define i32 @bad(double %x) {
  %1 = fptoui double %x to i32
  ret i32 %1

; CHECK: fctidz [[REG0:[0-9]+]], 1
; CHECK: stfd [[REG0]], [[OFF:.*]](1)
; CHECK: lwz {{[0-9]*}}, [[OFF]](1)
; GENERIC: xscvdpuxws [[REG0:[0-9]+]], 1
; GENERIC: mffprwz  {{[0-9]*}}, [[REG0]]
}

define i32 @bad1(float %x) {
entry:
  %0 = fptosi float %x to i32
  ret i32 %0

; CHECK: fctiwz [[REG0:[0-9]+]], 1
; CHECK: stfd [[REG0]], [[OFF:.*]](1)
; CHECK: lwa {{[0-9]*}}, [[OFF]](1)
; GENERIC: xscvdpsxws [[REG0:[0-9]+]], 1
; GENERIC: mffprwz  {{[0-9]*}}, [[REG0]]
}
