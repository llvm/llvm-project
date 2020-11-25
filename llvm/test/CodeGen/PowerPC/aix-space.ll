; RUN: llc -verify-machineinstrs -O0 -mcpu=pwr7 -mattr=-altivec -mtriple powerpc-ibm-aix-xcoff < %s | FileCheck %s

@a = common global double 0.000000e+00, align 8

; Get some constants into the constant pool that need spacing for alignment
define void @e() {
entry:
  %0 = load double, double* @a, align 8
  %mul = fmul double 1.500000e+00, %0
  store double %mul, double* @a, align 8
  %mul1 = fmul double 0x3F9C71C71C71C71C, %0
  store double %mul1, double* @a, align 8
  ret void
}

; CHECK:      .space 4
; CHECK-NOT:  .zero
