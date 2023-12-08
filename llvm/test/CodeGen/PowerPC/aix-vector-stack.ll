; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec \
; RUN:     -vec-extabi -mtriple powerpc-ibm-aix-xcoff < %s | \
; RUN: FileCheck %s --check-prefix=32BIT

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=+altivec \
; RUN:     -vec-extabi -mtriple powerpc64-ibm-aix-xcoff < %s | \
; RUN: FileCheck %s --check-prefix=64BIT

define dso_local <4 x i32> @vec_callee_stack(<4 x i32> %vr2, <4 x i32> %vr3, <4 x i32> %vr4, <4 x i32> %vr5, <4 x i32> %vr6, <4 x i32> %vr7, <4 x i32> %vr8, <4 x i32> %vr9, <4 x i32> %vr10, <4 x i32> %vr11, <4 x i32> %vr12, <4 x i32> %vr13, <4 x i32> %vSpill1, <4 x i32> %vSpill2) {
entry:
  ret <4 x i32> %vSpill2
}

; 32BIT:       addi [[SCRATCH:[0-9]+]], 1, 48
; 32BIT-NEXT:  lxvw4x 34, 0, [[SCRATCH]]

; 64BIT:       addi [[SCRATCH:[0-9]+]], 1, 64
; 64BIT-NEXT:  lxvw4x 34, 0, [[SCRATCH]]
