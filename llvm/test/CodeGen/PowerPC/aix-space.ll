; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr7 -mattr=-altivec -mtriple powerpc64-ibm-aix-xcoff < %s | FileCheck %s

@vi = global <4 x i32> zeroinitializer, section "custom_section", align 16
@f = global float 0x400921E480000000, section "custom_section", align 16

; CHECK:      .space 16
; CHECK-NOT:  .zero
