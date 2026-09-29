; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=-avx,+sse2 -fast-isel -fast-isel-abort=1 | FileCheck %s --check-prefix=SSE
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=-avx2,+avx -fast-isel -fast-isel-abort=1 | FileCheck %s --check-prefix=AVX

define i32 @cvt_test1(float %a) {
; SSE-LABEL: cvt_test1
; SSE:       cvttss2si %xmm0, %eax
; AVX-LABEL: cvt_test1
; AVX:       vcvttss2si %xmm0, %eax
  %1 = insertelement <4 x float> undef, float %a, i32 0
  %2 = insertelement <4 x float> %1, float 0.000000e+00, i32 1
  %3 = insertelement <4 x float> %2, float 0.000000e+00, i32 2
  %4 = insertelement <4 x float> %3, float 0.000000e+00, i32 3
  %5 = call i32 @llvm.x86.sse.cvttss2si(<4 x float> %4)
  ret i32 %5
}
declare i32 @llvm.x86.sse.cvttss2si(<4 x float>) nounwind readnone

define i64 @cvt_test2(float %a) {
; SSE-LABEL: cvt_test2
; SSE:       cvttss2si %xmm0, %rax
; AVX-LABEL: cvt_test2
; AVX:       vcvttss2si %xmm0, %rax
  %1 = insertelement <4 x float> undef, float %a, i32 0
  %2 = insertelement <4 x float> %1, float 0.000000e+00, i32 1
  %3 = insertelement <4 x float> %2, float 0.000000e+00, i32 2
  %4 = insertelement <4 x float> %3, float 0.000000e+00, i32 3
  %5 = call i64 @llvm.x86.sse.cvttss2si64(<4 x float> %4)
  ret i64 %5
}
declare i64 @llvm.x86.sse.cvttss2si64(<4 x float>) nounwind readnone

define i32 @cvt_test3(double %a) {
; SSE-LABEL: cvt_test3
; SSE:       cvttsd2si %xmm0, %eax
; AVX-LABEL: cvt_test3
; AVX:       vcvttsd2si %xmm0, %eax
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 0.000000e+00, i32 1
  %3 = call i32 @llvm.x86.sse2.cvttsd2si(<2 x double> %2)
  ret i32 %3
}
declare i32 @llvm.x86.sse2.cvttsd2si(<2 x double>) nounwind readnone

define i64 @cvt_test4(double %a) {
; SSE-LABEL: cvt_test4
; SSE:       cvttsd2si %xmm0, %rax
; AVX-LABEL: cvt_test4
; AVX:       vcvttsd2si %xmm0, %rax
  %1 = insertelement <2 x double> undef, double %a, i32 0
  %2 = insertelement <2 x double> %1, double 0.000000e+00, i32 1
  %3 = call i64 @llvm.x86.sse2.cvttsd2si64(<2 x double> %2)
  ret i64 %3
}
declare i64 @llvm.x86.sse2.cvttsd2si64(<2 x double>) nounwind readnone
