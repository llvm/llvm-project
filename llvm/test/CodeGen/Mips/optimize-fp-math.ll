; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=32
; RUN: llc -march=mips64el -mcpu=mips4 < %s | FileCheck %s -check-prefix=64
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s -check-prefix=64

; 32-LABEL: test_sqrtf_float_:
; 32: sqrt.s $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 32: c.un.s $f[[R0]], $f[[R0]]
; 64-LABEL: test_sqrtf_float_:
; 64: sqrt.s $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 64: c.un.s $f[[R0]], $f[[R0]]

define float @test_sqrtf_float_(float %a) {
entry:
  %call = tail call float @sqrtf(float %a)
  ret float %call
}

declare float @sqrtf(float)

; 32-LABEL: test_sqrt_double_:
; 32: sqrt.d $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 32: c.un.d $f[[R0]], $f[[R0]]
; 64-LABEL: test_sqrt_double_:
; 64: sqrt.d $f[[R0:[0-9]+]], $f{{[0-9]+}}
; 64: c.un.d $f[[R0]], $f[[R0]]

define double @test_sqrt_double_(double %a) {
entry:
  %call = tail call double @sqrt(double %a)
  ret double %call
}

declare double @sqrt(double)
