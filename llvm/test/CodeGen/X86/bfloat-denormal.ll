; RUN: llc < %s -mtriple=x86_64-linux-gnu -mattr=+avx512bf16,+avx512vl -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,AVX512BF16
; RUN: llc < %s -mtriple=x86_64-linux-gnu -mattr=+avxneconvert -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,AVXNECONVERT

; VCVTNEPS2BF16 ignores MXCSR and treats input denormals as signed zero. Only
; use it when the function's f32 input denormal mode has the same behavior.
; Otherwise round with integer arithmetic instead of calling __truncsfbf2.

define bfloat @fptrunc_default(float %x) nounwind {
; CHECK-LABEL: fptrunc_default:
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         vucomiss %xmm0, %xmm0
; CHECK:         shrl $16
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         retq
  %r = fptrunc float %x to bfloat
  ret bfloat %r
}

define bfloat @fptrunc_preservesign(float %x) nounwind denormal_fpenv(preservesign) {
; AVX512BF16-LABEL: fptrunc_preservesign:
; AVX512BF16:       vcvtneps2bf16 %xmm0, %xmm0
; AVX512BF16:       retq
;
; AVXNECONVERT-LABEL: fptrunc_preservesign:
; AVXNECONVERT:       {vex} vcvtneps2bf16 %xmm0, %xmm0
; AVXNECONVERT:       retq
  %r = fptrunc float %x to bfloat
  ret bfloat %r
}

; Positive-zero mode cannot use an instruction that preserves the input sign.
define bfloat @fptrunc_positivezero(float %x) nounwind denormal_fpenv(positivezero) {
; CHECK-LABEL: fptrunc_positivezero:
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         vucomiss %xmm0, %xmm0
; CHECK:         shrl $16
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         retq
  %r = fptrunc float %x to bfloat
  ret bfloat %r
}

define bfloat @fptrunc_dynamic(float %x) nounwind denormal_fpenv(dynamic) {
; CHECK-LABEL: fptrunc_dynamic:
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         vucomiss %xmm0, %xmm0
; CHECK:         shrl $16
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         retq
  %r = fptrunc float %x to bfloat
  ret bfloat %r
}

; Only the input mode matters: f32 and bf16 have the same exponent range, so a
; normal f32 cannot round to a bf16 denormal.
define bfloat @fptrunc_output_preservesign(float %x) nounwind denormal_fpenv(preservesign|ieee) {
; CHECK-LABEL: fptrunc_output_preservesign:
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         vucomiss %xmm0, %xmm0
; CHECK:         shrl $16
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         retq
  %r = fptrunc float %x to bfloat
  ret bfloat %r
}

define bfloat @fptrunc_input_preservesign(float %x) nounwind denormal_fpenv(ieee|preservesign) {
; AVX512BF16-LABEL: fptrunc_input_preservesign:
; AVX512BF16:       vcvtneps2bf16 %xmm0, %xmm0
; AVX512BF16:       retq
;
; AVXNECONVERT-LABEL: fptrunc_input_preservesign:
; AVXNECONVERT:       {vex} vcvtneps2bf16 %xmm0, %xmm0
; AVXNECONVERT:       retq
  %r = fptrunc float %x to bfloat
  ret bfloat %r
}

define <8 x bfloat> @fptrunc_v8_default(<8 x float> %x) nounwind {
; CHECK-LABEL: fptrunc_v8_default:
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         vcmpunordps %ymm0, %ymm0
; CHECK:         vpsrld $16
; CHECK-NOT:     vcvtneps2bf16
; CHECK-NOT:     __truncsfbf2
; CHECK:         retq
  %r = fptrunc <8 x float> %x to <8 x bfloat>
  ret <8 x bfloat> %r
}

define <8 x bfloat> @fptrunc_v8_preservesign(<8 x float> %x) nounwind denormal_fpenv(preservesign) {
; AVX512BF16-LABEL: fptrunc_v8_preservesign:
; AVX512BF16:       vcvtneps2bf16 %ymm0, %xmm0
; AVX512BF16:       retq
;
; AVXNECONVERT-LABEL: fptrunc_v8_preservesign:
; AVXNECONVERT:       {vex} vcvtneps2bf16 %ymm0, %xmm0
; AVXNECONVERT:       retq
  %r = fptrunc <8 x float> %x to <8 x bfloat>
  ret <8 x bfloat> %r
}
