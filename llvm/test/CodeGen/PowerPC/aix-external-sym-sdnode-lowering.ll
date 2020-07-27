; RUN: llc -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN: -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=32BIT %s

; RUN: llc -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=64BIT %s

define double @call_ceil(double %n) {
entry:
  %0 = call double @llvm.ceil.f64(double %n)
  ret double %0
}

declare double @llvm.ceil.f64(double)

; 32BIT: BL_NOP &.ceil
; 64BIT: BL8_NOP &.ceil
