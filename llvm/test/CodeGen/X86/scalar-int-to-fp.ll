; Verify that scalar integer conversions to FP compile successfully
; (at one time long double failed with avx512f), and that reasonable
; instruction sequences are selected based on subtarget features.
; Due to the plethora of reasonable sequences we just check for
; one key instruction, usually a cvt or fild, allowing the test
; to be relatively easily updated when sequences are improved.
;
; RUN: llc < %s -mtriple=i386-unknown-unknown     -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512_32
; RUN: llc < %s -mtriple=x86_64-unknown-unknown   -mattr=+avx512f | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512_64
; RUN: llc < %s -mtriple=i386-unknown-unknown     -mattr=+sse2    | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2_32
; RUN: llc < %s -mtriple=x86_64-unknown-unknown   -mattr=+sse2    | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2_64
; RUN: llc < %s -mtriple=i386-unknown-unknown     -mattr=-sse     | FileCheck %s --check-prefix=CHECK --check-prefix=X87

; CHECK-LABEL: u32_to_f
; AVX512_32: vcvtusi2ssl
; AVX512_64: vcvtusi2ssl
; SSE2_32: cvtsd2ss
; SSE2_64: cvtsi2ssq
; X87: fildll
define float @u32_to_f(i32 %a) nounwind {
  %r = uitofp i32 %a to float
  ret float %r
}

; CHECK-LABEL: s32_to_f
; AVX512_32: vcvtsi2ssl
; AVX512_64: vcvtsi2ssl
; SSE2_32: cvtsi2ssl
; SSE2_64: cvtsi2ssl
; X87: fildl
define float @s32_to_f(i32 %a) nounwind {
  %r = sitofp i32 %a to float
  ret float %r
}

; CHECK-LABEL: u32_to_d
; AVX512_32: vcvtusi2sdl
; AVX512_64: vcvtusi2sdl
; SSE2_32: subsd
; SSE2_64: cvtsi2sdq
; X87: fildll
define double @u32_to_d(i32 %a) nounwind {
  %r = uitofp i32 %a to double
  ret double %r
}

; CHECK-LABEL: s32_to_d
; AVX512_32: vcvtsi2sdl
; AVX512_64: vcvtsi2sdl
; SSE2_32: cvtsi2sdl
; SSE2_64: cvtsi2sdl
; X87: fildl
define double @s32_to_d(i32 %a) nounwind {
  %r = sitofp i32 %a to double
  ret double %r
}

; CHECK-LABEL: u32_to_x
; AVX512_32: vsubsd
; AVX512_64: vsubsd
; SSE2_32: subsd
; SSE2_64: fildll
; X87: fildll
define x86_fp80 @u32_to_x(i32 %a) nounwind {
  %r = uitofp i32 %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: s32_to_x
; CHECK: fildl
define x86_fp80 @s32_to_x(i32 %a) nounwind {
  %r = sitofp i32 %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: u64_to_f
; AVX512_32: fildll
; AVX512_64: vcvtusi2ssq
; SSE2_32: fildll
; SSE2_64: cvtsi2ssq
; X87: fildll
define float @u64_to_f(i64 %a) nounwind {
  %r = uitofp i64 %a to float
  ret float %r
}

; CHECK-LABEL: s64_to_f
; AVX512_32: fildll
; AVX512_64: vcvtsi2ssq
; SSE2_32: fildll
; SSE2_64: cvtsi2ssq
; X87: fildll
define float @s64_to_f(i64 %a) nounwind {
  %r = sitofp i64 %a to float
  ret float %r
}

; CHECK-LABEL: u64_to_d
; AVX512_32: vpunpckldq
; AVX512_64: vcvtusi2sdq
; SSE2_32: punpckldq
; SSE2_64: punpckldq
; X87: fildll
define double @u64_to_d(i64 %a) nounwind {
  %r = uitofp i64 %a to double
  ret double %r
}

; CHECK-LABEL: s64_to_d
; AVX512_32: fildll
; AVX512_64: vcvtsi2sdq
; SSE2_32: fildll
; SSE2_64: cvtsi2sdq
; X87: fildll
define double @s64_to_d(i64 %a) nounwind {
  %r = sitofp i64 %a to double
  ret double %r
}

; CHECK-LABEL: u64_to_x
; CHECK: fildll
define x86_fp80 @u64_to_x(i64 %a) nounwind {
  %r = uitofp i64 %a to x86_fp80
  ret x86_fp80 %r
}

; CHECK-LABEL: s64_to_x
; CHECK: fildll
define x86_fp80 @s64_to_x(i64 %a) nounwind {
  %r = sitofp i64 %a to x86_fp80
  ret x86_fp80 %r
}
