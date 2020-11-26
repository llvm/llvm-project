; RUN:  llc < %s -mtriple powerpc64-ibm-aix-xcoff -vec-extabi -mcpu=pwr8 2>&1
;       | FileCheck %s --check-prefix=ASM64
; RUN:  llc < %s -mtriple powerpc-ibm-aix-xcoff -vec-extabi -mcpu=pwr8 2>&1
;       | FileCheck %s --check-prefix=ASM32

@f1 = global float 0.000000e+00, align 4

define void @call_test_vararg() {
entry:
  %0 = load float, float* @f1, align 4
  %conv = fpext float %0 to double
  call void (i32, ...) @test_vararg(i32 42, double %conv, float %0)
  ret void
}

declare void @test_vararg(i32, ...)


; ASM64:           xscvdpspn
; ASM64:           mffprd
; ASM64:           xxsldwi
; ASM64:           mffprwz


; ASM32:           lfsx
; ASM32:           fmr
; ASM32:           stfs
; ASM32:           lwz
; ASM32:           stfd
