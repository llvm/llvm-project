; RUN: llc -march=mipsel < %s | FileCheck %s -check-prefix=32
; RUN: llc -march=mips64el -mcpu=mips4 < %s | FileCheck %s -check-prefix=64
; RUN: llc -march=mips64el -mcpu=mips64 < %s | FileCheck %s -check-prefix=64

@i1 = global [3 x i32] [i32 1, i32 2, i32 3], align 4
@i3 = common global ptr null, align 4

; 32-LABEL: test_float_int_:
; 32: mtc1 ${{[0-9]+}}, $f[[R0:[0-9]+]]
; 32: cvt.s.w $f{{[0-9]+}}, $f[[R0]]

define float @test_float_int_(i32 %a) {
entry:
  %conv = sitofp i32 %a to float
  ret float %conv
}

; 32-LABEL: test_double_int_:
; 32: mtc1 ${{[0-9]+}}, $f[[R0:[0-9]+]]
; 32: cvt.d.w $f{{[0-9]+}}, $f[[R0]]
; 64-LABEL: test_double_int_:
; 64: mtc1 ${{[0-9]+}}, $f[[R0:[0-9]+]]
; 64: cvt.d.w $f{{[0-9]+}}, $f[[R0]]

define double @test_double_int_(i32 %a) {
entry:
  %conv = sitofp i32 %a to double
  ret double %conv
}

; 64-LABEL: test_float_LL_:
; 64: dmtc1 ${{[0-9]+}}, $f[[R0:[0-9]+]]
; 64: cvt.s.l $f{{[0-9]+}}, $f[[R0]]

define float @test_float_LL_(i64 %a) {
entry:
  %conv = sitofp i64 %a to float
  ret float %conv
}

; 64-LABEL: test_double_LL_:
; 64: dmtc1 ${{[0-9]+}}, $f[[R0:[0-9]+]]
; 64: cvt.d.l $f{{[0-9]+}}, $f[[R0]]

define double @test_double_LL_(i64 %a) {
entry:
  %conv = sitofp i64 %a to double
  ret double %conv
}
