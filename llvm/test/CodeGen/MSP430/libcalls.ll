; RUN: llc -O0 < %s | FileCheck %s

target datalayout = "e-p:16:16:16-i8:8:8-i16:16:16-i32:16:32-n8:16-a0:16:16"
target triple = "msp430---elf"

@g_double = global double 123.0, align 8
@g_float = global float 123.0, align 8
@g_i32 = global i32 123, align 8
@g_i64 = global i64 456, align 8
@g_i16 = global i16 789, align 8

define float @d2f() #0 {
entry:
; CHECK: d2f:

; CHECK: call #__mspabi_cvtdf
  %0 = load volatile double, double* @g_double, align 8
  %1 = fptrunc double %0 to float

  ret float %1
}

define double @f2d() #0 {
entry:
; CHECK: f2d:

; CHECK: call #__mspabi_cvtfd
  %0 = load volatile float, float* @g_float, align 8
  %1 = fpext float %0 to double

  ret double %1
}

define i32 @d2l() #0 {
entry:
; CHECK: d2l:

; CHECK: call #__mspabi_fixdli
  %0 = load volatile double, double* @g_double, align 8
  %1 = fptosi double %0 to i32

  ret i32 %1
}

define i64 @d2ll() #0 {
entry:
; CHECK: d2ll:

; CHECK: call #__mspabi_fixdlli
  %0 = load volatile double, double* @g_double, align 8
  %1 = fptosi double %0 to i64

  ret i64 %1
}

define i32 @d2ul() #0 {
entry:
; CHECK: d2ul:

; CHECK: call #__mspabi_fixdul
  %0 = load volatile double, double* @g_double, align 8
  %1 = fptoui double %0 to i32

  ret i32 %1
}

define i64 @d2ull() #0 {
entry:
; CHECK: d2ull:

; CHECK: call #__mspabi_fixdull
  %0 = load volatile double, double* @g_double, align 8
  %1 = fptoui double %0 to i64

  ret i64 %1
}

define i32 @f2l() #0 {
entry:
; CHECK: f2l:

; CHECK: call #__mspabi_fixfli
  %0 = load volatile float, float* @g_float, align 8
  %1 = fptosi float %0 to i32

  ret i32 %1
}

define i64 @f2ll() #0 {
entry:
; CHECK: f2ll:

; CHECK: call #__mspabi_fixflli
  %0 = load volatile float, float* @g_float, align 8
  %1 = fptosi float %0 to i64

  ret i64 %1
}

define i32 @f2ul() #0 {
entry:
; CHECK: f2ul:

; CHECK: call #__mspabi_fixful
  %0 = load volatile float, float* @g_float, align 8
  %1 = fptoui float %0 to i32

  ret i32 %1
}

define i64 @f2ull() #0 {
entry:
; CHECK: f2ull:

; CHECK: call #__mspabi_fixfull
  %0 = load volatile float, float* @g_float, align 8
  %1 = fptoui float %0 to i64

  ret i64 %1
}

define double @l2d() #0 {
entry:
; CHECK: l2d:

; CHECK: call #__mspabi_fltlid
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = sitofp i32 %0 to double

  ret double %1
}

define double @ll2d() #0 {
entry:
; CHECK: ll2d:

; CHECK: call #__mspabi_fltllid
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = sitofp i64 %0 to double

  ret double %1
}

define double @ul2d() #0 {
entry:
; CHECK: ul2d:

; CHECK: call #__mspabi_fltuld
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = uitofp i32 %0 to double

  ret double %1
}

define double @ull2d() #0 {
entry:
; CHECK: ull2d:

; CHECK: call #__mspabi_fltulld
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = uitofp i64 %0 to double

  ret double %1
}

define float @l2f() #0 {
entry:
; CHECK: l2f:

; CHECK: call #__mspabi_fltlif
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = sitofp i32 %0 to float

  ret float %1
}

define float @ll2f() #0 {
entry:
; CHECK: ll2f:

; CHECK: call #__mspabi_fltllif
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = sitofp i64 %0 to float

  ret float %1
}

define float @ul2f() #0 {
entry:
; CHECK: ul2f:

; CHECK: call #__mspabi_fltulf
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = uitofp i32 %0 to float

  ret float %1
}

define float @ull2f() #0 {
entry:
; CHECK: ull2f:

; CHECK: call #__mspabi_fltullf
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = uitofp i64 %0 to float

  ret float %1
}

define i1 @cmpd_oeq() #0 {
entry:
; CHECK: cmpd_oeq:

; CHECK: call #__mspabi_cmpd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fcmp oeq double %0, 123.0

  ret i1 %1
}

define i1 @cmpd_une() #0 {
entry:
; CHECK: cmpd_une:

; CHECK: call #__mspabi_cmpd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fcmp une double %0, 123.0

  ret i1 %1
}

define i1 @cmpd_oge() #0 {
entry:
; CHECK: cmpd_oge:

; CHECK: call #__mspabi_cmpd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fcmp oge double %0, 123.0

  ret i1 %1
}

define i1 @cmpd_olt() #0 {
entry:
; CHECK: cmpd_olt:

; CHECK: call #__mspabi_cmpd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fcmp olt double %0, 123.0

  ret i1 %1
}

define i1 @cmpd_ole() #0 {
entry:
; CHECK: cmpd_ole:

; CHECK: call #__mspabi_cmpd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fcmp ole double %0, 123.0

  ret i1 %1
}

define i1 @cmpd_ogt() #0 {
entry:
; CHECK: cmpd_ogt:

; CHECK: call #__mspabi_cmpd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fcmp ogt double %0, 123.0

  ret i1 %1
}

define i1 @cmpf_oeq() #0 {
entry:
; CHECK: cmpf_oeq:

; CHECK: call #__mspabi_cmpf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fcmp oeq float %0, 123.0

  ret i1 %1
}

define i1 @cmpf_une() #0 {
entry:
; CHECK: cmpf_une:

; CHECK: call #__mspabi_cmpf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fcmp une float %0, 123.0

  ret i1 %1
}

define i1 @cmpf_oge() #0 {
entry:
; CHECK: cmpf_oge:

; CHECK: call #__mspabi_cmpf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fcmp oge float %0, 123.0

  ret i1 %1
}

define i1 @cmpf_olt() #0 {
entry:
; CHECK: cmpf_olt:

; CHECK: call #__mspabi_cmpf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fcmp olt float %0, 123.0

  ret i1 %1
}

define i1 @cmpf_ole() #0 {
entry:
; CHECK: cmpf_ole:

; CHECK: call #__mspabi_cmpf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fcmp ole float %0, 123.0

  ret i1 %1
}

define i1 @cmpf_ogt() #0 {
entry:
; CHECK: cmpf_ogt:

; CHECK: call #__mspabi_cmpf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fcmp ogt float %0, 123.0

  ret i1 %1
}

define double @addd() #0 {
entry:
; CHECK: addd:

; CHECK: call #__mspabi_addd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fadd double %0, 123.0

  ret double %1
}

define float @addf() #0 {
entry:
; CHECK: addf:

; CHECK: call #__mspabi_addf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fadd float %0, 123.0

  ret float %1
}

define double @divd() #0 {
entry:
; CHECK: divd:

; CHECK: call #__mspabi_divd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fdiv double %0, 123.0

  ret double %1
}

define float @divf() #0 {
entry:
; CHECK: divf:

; CHECK: call #__mspabi_divf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fdiv float %0, 123.0

  ret float %1
}

define double @mpyd() #0 {
entry:
; CHECK: mpyd:

; CHECK: call #__mspabi_mpyd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fmul double %0, 123.0

  ret double %1
}

define float @mpyf() #0 {
entry:
; CHECK: mpyf:

; CHECK: call #__mspabi_mpyf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fmul float %0, 123.0

  ret float %1
}

define double @subd() #0 {
entry:
; CHECK: subd:

; CHECK: call #__mspabi_subd
  %0 = load volatile double, double* @g_double, align 8
  %1 = fsub double %0, %0

  ret double %1
}

define float @subf() #0 {
entry:
; CHECK: subf:

; CHECK: call #__mspabi_subf
  %0 = load volatile float, float* @g_float, align 8
  %1 = fsub float %0, %0

  ret float %1
}

define i16 @divi() #0 {
entry:
; CHECK: divi:

; CHECK: call #__mspabi_divi
  %0 = load volatile i16, i16* @g_i16, align 8
  %1 = sdiv i16 %0, %0

  ret i16 %1
}

define i32 @divli() #0 {
entry:
; CHECK: divli:

; CHECK: call #__mspabi_divli
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = sdiv i32 %0, %0

  ret i32 %1
}

define i64 @divlli() #0 {
entry:
; CHECK: divlli:

; CHECK: call #__mspabi_divlli
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = sdiv i64 %0, %0

  ret i64 %1
}

define i16 @divu() #0 {
entry:
; CHECK: divu:

; CHECK: call #__mspabi_divu
  %0 = load volatile i16, i16* @g_i16, align 8
  %1 = udiv i16 %0, %0

  ret i16 %1
}

define i32 @divul() #0 {
entry:
; CHECK: divul:

; CHECK: call #__mspabi_divul
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = udiv i32 %0, %0

  ret i32 %1
}

define i64 @divull() #0 {
entry:
; CHECK: divull:

; CHECK: call #__mspabi_divull
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = udiv i64 %0, %0

  ret i64 %1
}

define i16 @remi() #0 {
entry:
; CHECK: remi:

; CHECK: call #__mspabi_remi
  %0 = load volatile i16, i16* @g_i16, align 8
  %1 = srem i16 %0, %0

  ret i16 %1
}

define i32 @remli() #0 {
entry:
; CHECK: remli:

; CHECK: call #__mspabi_remli
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = srem i32 %0, %0

  ret i32 %1
}

define i64 @remlli() #0 {
entry:
; CHECK: remlli:

; CHECK: call #__mspabi_remlli
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = srem i64 %0, %0

  ret i64 %1
}

define i16 @remu() #0 {
entry:
; CHECK: remu:

; CHECK: call #__mspabi_remu
  %0 = load volatile i16, i16* @g_i16, align 8
  %1 = urem i16 %0, %0

  ret i16 %1
}

define i32 @remul() #0 {
entry:
; CHECK: remul:

; CHECK: call #__mspabi_remul
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = urem i32 %0, %0

  ret i32 %1
}

define i64 @remull() #0 {
entry:
; CHECK: remull:

; CHECK: call #__mspabi_remull
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = urem i64 %0, %0

  ret i64 %1
}

define i16 @mpyi() #0 {
entry:
; CHECK: mpyi:

; CHECK: call #__mspabi_mpyi
  %0 = load volatile i16, i16* @g_i16, align 8
  %1 = mul i16 %0, %0

  ret i16 %1
}

define i32 @mpyli() #0 {
entry:
; CHECK: mpyli:

; CHECK: call #__mspabi_mpyl
  %0 = load volatile i32, i32* @g_i32, align 8
  %1 = mul i32 %0, %0

  ret i32 %1
}

define i64 @mpylli() #0 {
entry:
; CHECK: mpylli:

; CHECK: call #__mspabi_mpyll
  %0 = load volatile i64, i64* @g_i64, align 8
  %1 = mul i64 %0, %0

  ret i64 %1
}

attributes #0 = { nounwind }
