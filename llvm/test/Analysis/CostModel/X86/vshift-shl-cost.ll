; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+sse2 -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=SSE2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+sse4.1 -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=SSE41
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=AVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx2 -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=AVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+xop,+avx -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=XOP --check-prefix=XOPAVX
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+xop,+avx2 -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=XOP --check-prefix=XOPAVX2
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512dq -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512F
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BW
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512vl -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512VL
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512dq,+avx512vl -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512VL
; RUN: opt < %s -mtriple=x86_64-unknown-linux-gnu -mattr=+avx512f,+avx512bw,+avx512vl -cost-model -analyze | FileCheck %s --check-prefix=CHECK --check-prefix=AVX512 --check-prefix=AVX512BWVL

; Verify the cost of vector shift left instructions.

;
;
; Variable Shifts
;

define <2 x i64> @var_shift_v2i64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v2i64':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <2 x i64> %a, %b
  ret <2 x i64> %shift
}

define <4 x i64> @var_shift_v4i64(<4 x i64> %a, <4 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v4i64':
; SSE2: Found an estimated cost of 8 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <4 x i64> %a, %b
  ret <4 x i64> %shift
}

define <8 x i64> @var_shift_v8i64(<8 x i64> %a, <8 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v8i64':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 16 for instruction:   %shift
; AVX: Found an estimated cost of 16 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <8 x i64> %a, %b
  ret <8 x i64> %shift
}

define <4 x i32> @var_shift_v4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v4i32':
; SSE2: Found an estimated cost of 10 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <4 x i32> %a, %b
  ret <4 x i32> %shift
}

define <8 x i32> @var_shift_v8i32(<8 x i32> %a, <8 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v8i32':
; SSE2: Found an estimated cost of 20 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <8 x i32> %a, %b
  ret <8 x i32> %shift
}

define <16 x i32> @var_shift_v16i32(<16 x i32> %a, <16 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v16i32':
; SSE2: Found an estimated cost of 40 for instruction:   %shift
; SSE41: Found an estimated cost of 16 for instruction:   %shift
; AVX: Found an estimated cost of 16 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <16 x i32> %a, %b
  ret <16 x i32> %shift
}

define <8 x i16> @var_shift_v8i16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v8i16':
; SSE2: Found an estimated cost of 32 for instruction:   %shift
; SSE41: Found an estimated cost of 14 for instruction:   %shift
; AVX: Found an estimated cost of 14 for instruction:   %shift
; AVX2: Found an estimated cost of 14 for instruction:   %shift
; AVX512: Found an estimated cost of 14 for instruction:   %shift
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <8 x i16> %a, %b
  ret <8 x i16> %shift
}

define <16 x i16> @var_shift_v16i16(<16 x i16> %a, <16 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v16i16':
; SSE2: Found an estimated cost of 64 for instruction:   %shift
; SSE41: Found an estimated cost of 28 for instruction:   %shift
; AVX: Found an estimated cost of 28 for instruction:   %shift
; AVX2: Found an estimated cost of 10 for instruction:   %shift
; AVX512: Found an estimated cost of 10 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <16 x i16> %a, %b
  ret <16 x i16> %shift
}

define <32 x i16> @var_shift_v32i16(<32 x i16> %a, <32 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v32i16':
; SSE2: Found an estimated cost of 128 for instruction:   %shift
; SSE41: Found an estimated cost of 56 for instruction:   %shift
; AVX: Found an estimated cost of 56 for instruction:   %shift
; AVX2: Found an estimated cost of 20 for instruction:   %shift
; AVX512F: Found an estimated cost of 20 for instruction:   %shift
; AVX512BW: Found an estimated cost of 1 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %shift = shl <32 x i16> %a, %b
  ret <32 x i16> %shift
}

define <16 x i8> @var_shift_v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v16i8':
; SSE2: Found an estimated cost of 26 for instruction:   %shift
; SSE41: Found an estimated cost of 11 for instruction:   %shift
; AVX: Found an estimated cost of 11 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <16 x i8> %a, %b
  ret <16 x i8> %shift
}

define <32 x i8> @var_shift_v32i8(<32 x i8> %a, <32 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v32i8':
; SSE2: Found an estimated cost of 52 for instruction:   %shift
; SSE41: Found an estimated cost of 22 for instruction:   %shift
; AVX: Found an estimated cost of 22 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <32 x i8> %a, %b
  ret <32 x i8> %shift
}

define <64 x i8> @var_shift_v64i8(<64 x i8> %a, <64 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v64i8':
; SSE2: Found an estimated cost of 104 for instruction:   %shift
; SSE41: Found an estimated cost of 44 for instruction:   %shift
; AVX: Found an estimated cost of 44 for instruction:   %shift
; AVX2: Found an estimated cost of 22 for instruction:   %shift
; AVX512F: Found an estimated cost of 22 for instruction:   %shift
; AVX512BW: Found an estimated cost of 11 for instruction:   %shift
; AVX512VL: Found an estimated cost of 22 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %shift = shl <64 x i8> %a, %b
  ret <64 x i8> %shift
}

;
; Uniform Variable Shifts
;

define <2 x i64> @splatvar_shift_v2i64(<2 x i64> %a, <2 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v2i64':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <2 x i64> %b, <2 x i64> undef, <2 x i32> zeroinitializer
  %shift = shl <2 x i64> %a, %splat
  ret <2 x i64> %shift
}

define <4 x i64> @splatvar_shift_v4i64(<4 x i64> %a, <4 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v4i64':
; SSE2: Found an estimated cost of 8 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <4 x i64> %b, <4 x i64> undef, <4 x i32> zeroinitializer
  %shift = shl <4 x i64> %a, %splat
  ret <4 x i64> %shift
}

define <8 x i64> @splatvar_shift_v8i64(<8 x i64> %a, <8 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v8i64':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 16 for instruction:   %shift
; AVX: Found an estimated cost of 16 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %splat = shufflevector <8 x i64> %b, <8 x i64> undef, <8 x i32> zeroinitializer
  %shift = shl <8 x i64> %a, %splat
  ret <8 x i64> %shift
}

define <4 x i32> @splatvar_shift_v4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v4i32':
; SSE2: Found an estimated cost of 10 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <4 x i32> %b, <4 x i32> undef, <4 x i32> zeroinitializer
  %shift = shl <4 x i32> %a, %splat
  ret <4 x i32> %shift
}

define <8 x i32> @splatvar_shift_v8i32(<8 x i32> %a, <8 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v8i32':
; SSE2: Found an estimated cost of 20 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <8 x i32> %b, <8 x i32> undef, <8 x i32> zeroinitializer
  %shift = shl <8 x i32> %a, %splat
  ret <8 x i32> %shift
}

define <16 x i32> @splatvar_shift_v16i32(<16 x i32> %a, <16 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v16i32':
; SSE2: Found an estimated cost of 40 for instruction:   %shift
; SSE41: Found an estimated cost of 16 for instruction:   %shift
; AVX: Found an estimated cost of 16 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %splat = shufflevector <16 x i32> %b, <16 x i32> undef, <16 x i32> zeroinitializer
  %shift = shl <16 x i32> %a, %splat
  ret <16 x i32> %shift
}

define <8 x i16> @splatvar_shift_v8i16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v8i16':
; SSE2: Found an estimated cost of 32 for instruction:   %shift
; SSE41: Found an estimated cost of 14 for instruction:   %shift
; AVX: Found an estimated cost of 14 for instruction:   %shift
; AVX2: Found an estimated cost of 14 for instruction:   %shift
; AVX512: Found an estimated cost of 14 for instruction:   %shift
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <8 x i16> %b, <8 x i16> undef, <8 x i32> zeroinitializer
  %shift = shl <8 x i16> %a, %splat
  ret <8 x i16> %shift
}

define <16 x i16> @splatvar_shift_v16i16(<16 x i16> %a, <16 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v16i16':
; SSE2: Found an estimated cost of 64 for instruction:   %shift
; SSE41: Found an estimated cost of 28 for instruction:   %shift
; AVX: Found an estimated cost of 28 for instruction:   %shift
; AVX2: Found an estimated cost of 10 for instruction:   %shift
; AVX512: Found an estimated cost of 10 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %splat = shufflevector <16 x i16> %b, <16 x i16> undef, <16 x i32> zeroinitializer
  %shift = shl <16 x i16> %a, %splat
  ret <16 x i16> %shift
}

define <32 x i16> @splatvar_shift_v32i16(<32 x i16> %a, <32 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v32i16':
; SSE2: Found an estimated cost of 128 for instruction:   %shift
; SSE41: Found an estimated cost of 56 for instruction:   %shift
; AVX: Found an estimated cost of 56 for instruction:   %shift
; AVX2: Found an estimated cost of 20 for instruction:   %shift
; AVX512F: Found an estimated cost of 20 for instruction:   %shift
; AVX512BW: Found an estimated cost of 1 for instruction:   %shift
; AVX512VL: Found an estimated cost of 20 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 1 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %splat = shufflevector <32 x i16> %b, <32 x i16> undef, <32 x i32> zeroinitializer
  %shift = shl <32 x i16> %a, %splat
  ret <32 x i16> %shift
}

define <16 x i8> @splatvar_shift_v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v16i8':
; SSE2: Found an estimated cost of 26 for instruction:   %shift
; SSE41: Found an estimated cost of 11 for instruction:   %shift
; AVX: Found an estimated cost of 11 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <16 x i8> %b, <16 x i8> undef, <16 x i32> zeroinitializer
  %shift = shl <16 x i8> %a, %splat
  ret <16 x i8> %shift
}

define <32 x i8> @splatvar_shift_v32i8(<32 x i8> %a, <32 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v32i8':
; SSE2: Found an estimated cost of 52 for instruction:   %shift
; SSE41: Found an estimated cost of 22 for instruction:   %shift
; AVX: Found an estimated cost of 22 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %splat = shufflevector <32 x i8> %b, <32 x i8> undef, <32 x i32> zeroinitializer
  %shift = shl <32 x i8> %a, %splat
  ret <32 x i8> %shift
}

define <64 x i8> @splatvar_shift_v64i8(<64 x i8> %a, <64 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v64i8':
; SSE2: Found an estimated cost of 104 for instruction:   %shift
; SSE41: Found an estimated cost of 44 for instruction:   %shift
; AVX: Found an estimated cost of 44 for instruction:   %shift
; AVX2: Found an estimated cost of 22 for instruction:   %shift
; AVX512F: Found an estimated cost of 22 for instruction:   %shift
; AVX512BW: Found an estimated cost of 11 for instruction:   %shift
; AVX512VL: Found an estimated cost of 22 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %splat = shufflevector <64 x i8> %b, <64 x i8> undef, <64 x i32> zeroinitializer
  %shift = shl <64 x i8> %a, %splat
  ret <64 x i8> %shift
}

;
; Constant Shifts
;

define <2 x i64> @constant_shift_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v2i64':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <2 x i64> %a, <i64 1, i64 7>
  ret <2 x i64> %shift
}

define <4 x i64> @constant_shift_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v4i64':
; SSE2: Found an estimated cost of 8 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <4 x i64> %a, <i64 1, i64 7, i64 15, i64 31>
  ret <4 x i64> %shift
}

define <8 x i64> @constant_shift_v8i64(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v8i64':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 16 for instruction:   %shift
; AVX: Found an estimated cost of 16 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <8 x i64> %a, <i64 1, i64 7, i64 15, i64 31, i64 1, i64 7, i64 15, i64 31>
  ret <8 x i64> %shift
}

define <4 x i32> @constant_shift_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v4i32':
; SSE2: Found an estimated cost of 6 for instruction:   %shift
; SSE41: Found an estimated cost of 1 for instruction:   %shift
; AVX: Found an estimated cost of 1 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <4 x i32> %a, <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i32> %shift
}

define <8 x i32> @constant_shift_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v8i32':
; SSE2: Found an estimated cost of 12 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <8 x i32> %a, <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <8 x i32> %shift
}

define <16 x i32> @constant_shift_v16i32(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v16i32':
; SSE2: Found an estimated cost of 24 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <16 x i32> %a, <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <16 x i32> %shift
}

define <8 x i16> @constant_shift_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v8i16':
; SSE2: Found an estimated cost of 1 for instruction:   %shift
; SSE41: Found an estimated cost of 1 for instruction:   %shift
; AVX: Found an estimated cost of 1 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <8 x i16> %a, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>
  ret <8 x i16> %shift
}

define <16 x i16> @constant_shift_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v16i16':
; SSE2: Found an estimated cost of 2 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <16 x i16> %a, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>
  ret <16 x i16> %shift
}

define <32 x i16> @constant_shift_v32i16(<32 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v32i16':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512F: Found an estimated cost of 2 for instruction:   %shift
; AVX512BW: Found an estimated cost of 1 for instruction:   %shift
; AVX512VL: Found an estimated cost of 2 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <32 x i16> %a, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>
  ret <32 x i16> %shift
}

define <16 x i8> @constant_shift_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v16i8':
; SSE2: Found an estimated cost of 26 for instruction:   %shift
; SSE41: Found an estimated cost of 11 for instruction:   %shift
; AVX: Found an estimated cost of 11 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <16 x i8> %a, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>
  ret <16 x i8> %shift
}

define <32 x i8> @constant_shift_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v32i8':
; SSE2: Found an estimated cost of 52 for instruction:   %shift
; SSE41: Found an estimated cost of 22 for instruction:   %shift
; AVX: Found an estimated cost of 22 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <32 x i8> %a, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>
  ret <32 x i8> %shift
}

define <64 x i8> @constant_shift_v64i8(<64 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v64i8':
; SSE2: Found an estimated cost of 104 for instruction:   %shift
; SSE41: Found an estimated cost of 44 for instruction:   %shift
; AVX: Found an estimated cost of 44 for instruction:   %shift
; AVX2: Found an estimated cost of 22 for instruction:   %shift
; AVX512F: Found an estimated cost of 22 for instruction:   %shift
; AVX512BW: Found an estimated cost of 11 for instruction:   %shift
; AVX512VL: Found an estimated cost of 22 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %shift = shl <64 x i8> %a, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>
  ret <64 x i8> %shift
}

;
; Uniform Constant Shifts
;

define <2 x i64> @splatconstant_shift_v2i64(<2 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v2i64':
; SSE2: Found an estimated cost of 1 for instruction:   %shift
; SSE41: Found an estimated cost of 1 for instruction:   %shift
; AVX: Found an estimated cost of 1 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <2 x i64> %a, <i64 7, i64 7>
  ret <2 x i64> %shift
}

define <4 x i64> @splatconstant_shift_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v4i64':
; SSE2: Found an estimated cost of 2 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 2 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <4 x i64> %a, <i64 7, i64 7, i64 7, i64 7>
  ret <4 x i64> %shift
}

define <8 x i64> @splatconstant_shift_v8i64(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v8i64':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <8 x i64> %a, <i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7>
  ret <8 x i64> %shift
}

define <4 x i32> @splatconstant_shift_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v4i32':
; SSE2: Found an estimated cost of 1 for instruction:   %shift
; SSE41: Found an estimated cost of 1 for instruction:   %shift
; AVX: Found an estimated cost of 1 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <4 x i32> %a, <i32 5, i32 5, i32 5, i32 5>
  ret <4 x i32> %shift
}

define <8 x i32> @splatconstant_shift_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v8i32':
; SSE2: Found an estimated cost of 2 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 2 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <8 x i32> %a, <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <8 x i32> %shift
}

define <16 x i32> @splatconstant_shift_v16i32(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v16i32':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <16 x i32> %a, <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <16 x i32> %shift
}

define <8 x i16> @splatconstant_shift_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v8i16':
; SSE2: Found an estimated cost of 1 for instruction:   %shift
; SSE41: Found an estimated cost of 1 for instruction:   %shift
; AVX: Found an estimated cost of 1 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <8 x i16> %shift
}

define <16 x i16> @splatconstant_shift_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v16i16':
; SSE2: Found an estimated cost of 2 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 2 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = shl <16 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <16 x i16> %shift
}

define <32 x i16> @splatconstant_shift_v32i16(<32 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v32i16':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512F: Found an estimated cost of 2 for instruction:   %shift
; AVX512BW: Found an estimated cost of 1 for instruction:   %shift
; AVX512VL: Found an estimated cost of 2 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <32 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <32 x i16> %shift
}

define <16 x i8> @splatconstant_shift_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v16i8':
; SSE2: Found an estimated cost of 2 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 2 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 2 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <16 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <16 x i8> %shift
}

define <32 x i8> @splatconstant_shift_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v32i8':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = shl <32 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <32 x i8> %shift
}

define <64 x i8> @splatconstant_shift_v64i8(<64 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v64i8':
; SSE2: Found an estimated cost of 8 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 4 for instruction:   %shift
; AVX512F: Found an estimated cost of 4 for instruction:   %shift
; AVX512BW: Found an estimated cost of 2 for instruction:   %shift
; AVX512VL: Found an estimated cost of 4 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 4 for instruction:   %shift
  %shift = shl <64 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <64 x i8> %shift
}

;
; Special Cases
;

; We always emit a single pmullw in the case of v8i16 vector shifts by
; non-uniform constant.

define <8 x i16> @test1(<8 x i16> %a) {
  %shl = shl <8 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <8 x i16> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test1':
; CHECK: Found an estimated cost of 1 for instruction:   %shl


define <8 x i16> @test2(<8 x i16> %a) {
  %shl = shl <8 x i16> %a, <i16 0, i16 undef, i16 0, i16 0, i16 1, i16 undef, i16 -1, i16 1>
  ret <8 x i16> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test2':
; CHECK: Found an estimated cost of 1 for instruction:   %shl


; With SSE4.1, v4i32 shifts can be lowered into a single pmulld instruction.
; Make sure that the estimated cost is always 1 except for the case where
; we only have SSE2 support. With SSE2, we are forced to special lower the
; v4i32 mul as a 2x shuffle, 2x pmuludq, 2x shuffle.

define <4 x i32> @test3(<4 x i32> %a) {
  %shl = shl <4 x i32> %a, <i32 1, i32 -1, i32 2, i32 -3>
  ret <4 x i32> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test3':
; SSE2: Found an estimated cost of 6 for instruction:   %shl
; SSE41: Found an estimated cost of 1 for instruction:   %shl
; AVX: Found an estimated cost of 1 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOP: Found an estimated cost of 1 for instruction:   %shl


define <4 x i32> @test4(<4 x i32> %a) {
  %shl = shl <4 x i32> %a, <i32 0, i32 0, i32 1, i32 1>
  ret <4 x i32> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test4':
; SSE2: Found an estimated cost of 6 for instruction:   %shl
; SSE41: Found an estimated cost of 1 for instruction:   %shl
; AVX: Found an estimated cost of 1 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOP: Found an estimated cost of 1 for instruction:   %shl


; On AVX2 we are able to lower the following shift into a single
; vpsllvq. Therefore, the expected cost is only 1.
; In all other cases, this shift is scalarized as the target does not support
; vpsllv instructions.

define <2 x i64> @test5(<2 x i64> %a) {
  %shl = shl <2 x i64> %a, <i64 2, i64 3>
  ret <2 x i64> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test5':
; SSE2: Found an estimated cost of 4 for instruction:   %shl
; SSE41: Found an estimated cost of 4 for instruction:   %shl
; AVX: Found an estimated cost of 4 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOP: Found an estimated cost of 1 for instruction:   %shl


; v16i16 and v8i32 shift left by non-uniform constant are lowered into
; vector multiply instructions.  With AVX (but not AVX2), the vector multiply
; is lowered into a sequence of: 1 extract + 2 vpmullw + 1 insert.
;
; With AVX2, instruction vpmullw works with 256bit quantities and
; therefore there is no need to split the resulting vector multiply into
; a sequence of two multiply.
;
; With SSE2 and SSE4.1, the vector shift cost for 'test6' is twice
; the cost computed in the case of 'test1'. That is because the backend
; simply emits 2 pmullw with no extract/insert.


define <16 x i16> @test6(<16 x i16> %a) {
  %shl = shl <16 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <16 x i16> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test6':
; SSE2: Found an estimated cost of 2 for instruction:   %shl
; SSE41: Found an estimated cost of 2 for instruction:   %shl
; AVX: Found an estimated cost of 4 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOPAVX: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shl


; With SSE2 and SSE4.1, the vector shift cost for 'test7' is twice
; the cost computed in the case of 'test3'. That is because the multiply
; is type-legalized into two 4i32 vector multiply.

define <8 x i32> @test7(<8 x i32> %a) {
  %shl = shl <8 x i32> %a, <i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3>
  ret <8 x i32> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test7':
; SSE2: Found an estimated cost of 12 for instruction:   %shl
; SSE41: Found an estimated cost of 2 for instruction:   %shl
; AVX: Found an estimated cost of 4 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOPAVX: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shl


; On AVX2 we are able to lower the following shift into a single
; vpsllvq. Therefore, the expected cost is only 1.
; In all other cases, this shift is scalarized as the target does not support
; vpsllv instructions.

define <4 x i64> @test8(<4 x i64> %a) {
  %shl = shl <4 x i64> %a, <i64 1, i64 2, i64 3, i64 4>
  ret <4 x i64> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test8':
; SSE2: Found an estimated cost of 8 for instruction:   %shl
; SSE41: Found an estimated cost of 8 for instruction:   %shl
; AVX: Found an estimated cost of 8 for instruction:   %shl
; AVX2: Found an estimated cost of 1 for instruction:   %shl
; XOPAVX: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shl


; Same as 'test6', with the difference that the cost is double.

define <32 x i16> @test9(<32 x i16> %a) {
  %shl = shl <32 x i16> %a, <i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11, i16 1, i16 1, i16 2, i16 3, i16 7, i16 0, i16 9, i16 11>
  ret <32 x i16> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test9':
; SSE2: Found an estimated cost of 4 for instruction:   %shl
; SSE41: Found an estimated cost of 4 for instruction:   %shl
; AVX: Found an estimated cost of 8 for instruction:   %shl
; AVX2: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX: Found an estimated cost of 4 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shl


; Same as 'test7', except that now the cost is double.

define <16 x i32> @test10(<16 x i32> %a) {
  %shl = shl <16 x i32> %a, <i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3, i32 1, i32 1, i32 2, i32 3>
  ret <16 x i32> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test10':
; SSE2: Found an estimated cost of 24 for instruction:   %shl
; SSE41: Found an estimated cost of 4 for instruction:   %shl
; AVX: Found an estimated cost of 8 for instruction:   %shl
; AVX2: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX: Found an estimated cost of 4 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shl


; On AVX2 we are able to lower the following shift into a sequence of
; two vpsllvq instructions. Therefore, the expected cost is only 2.
; In all other cases, this shift is scalarized as we don't have vpsllv
; instructions.

define <8 x i64> @test11(<8 x i64> %a) {
  %shl = shl <8 x i64> %a, <i64 1, i64 1, i64 2, i64 3, i64 1, i64 1, i64 2, i64 3>
  ret <8 x i64> %shl
}
; CHECK: 'Cost Model Analysis' for function 'test11':
; SSE2: Found an estimated cost of 16 for instruction:   %shl
; SSE41: Found an estimated cost of 16 for instruction:   %shl
; AVX: Found an estimated cost of 16 for instruction:   %shl
; AVX2: Found an estimated cost of 2 for instruction:   %shl
; XOPAVX: Found an estimated cost of 4 for instruction:   %shl
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shl
