; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=skx | FileCheck --check-prefix=SKX %s


define <8 x i16> @extract_subvector128_v32i16(<32 x i16> %x) nounwind {
; SKX-LABEL: extract_subvector128_v32i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vextracti32x4 $2, %zmm0, %xmm0
; SKX-NEXT:    retq
  %r1 = shufflevector <32 x i16> %x, <32 x i16> undef, <8 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23>
  ret <8 x i16> %r1
}

define <8 x i16> @extract_subvector128_v32i16_first_element(<32 x i16> %x) nounwind {
; SKX-LABEL: extract_subvector128_v32i16_first_element:
; SKX:       ## BB#0:
; SKX-NEXT:    retq
  %r1 = shufflevector <32 x i16> %x, <32 x i16> undef, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  ret <8 x i16> %r1
}

define <16 x i8> @extract_subvector128_v64i8(<64 x i8> %x) nounwind {
; SKX-LABEL: extract_subvector128_v64i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vextracti32x4 $2, %zmm0, %xmm0
; SKX-NEXT:    retq
  %r1 = shufflevector <64 x i8> %x, <64 x i8> undef, <16 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38,i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47>
  ret <16 x i8> %r1
}

define <16 x i8> @extract_subvector128_v64i8_first_element(<64 x i8> %x) nounwind {
; SKX-LABEL: extract_subvector128_v64i8_first_element:
; SKX:       ## BB#0:
; SKX-NEXT:    retq
  %r1 = shufflevector <64 x i8> %x, <64 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i8> %r1
}


define <16 x i16> @extract_subvector256_v32i16(<32 x i16> %x) nounwind {
; SKX-LABEL: extract_subvector256_v32i16:
; SKX:       ## BB#0:
; SKX-NEXT:    vextracti64x4 $1, %zmm0, %ymm0
; SKX-NEXT:    retq
  %r1 = shufflevector <32 x i16> %x, <32 x i16> undef, <16 x i32> <i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <16 x i16> %r1
}

define <32 x i8> @extract_subvector256_v64i8(<64 x i8> %x) nounwind {
; SKX-LABEL: extract_subvector256_v64i8:
; SKX:       ## BB#0:
; SKX-NEXT:    vextracti64x4 $1, %zmm0, %ymm0
; SKX-NEXT:    retq
  %r1 = shufflevector <64 x i8> %x, <64 x i8> undef, <32 x i32> <i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 45, i32 46, i32 47, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60, i32 61, i32 62, i32 63>
  ret <32 x i8> %r1
}
