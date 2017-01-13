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

; Verify the cost of vector logical shift right instructions.

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
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <2 x i64> %a, %b
  ret <2 x i64> %shift
}

define <4 x i64> @var_shift_v4i64(<4 x i64> %a, <4 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v4i64':
; SSE2: Found an estimated cost of 8 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <4 x i64> %a, %b
  ret <4 x i64> %shift
}

define <8 x i64> @var_shift_v8i64(<8 x i64> %a, <8 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v8i64':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 16 for instruction:   %shift
; AVX: Found an estimated cost of 16 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <8 x i64> %a, %b
  ret <8 x i64> %shift
}

define <4 x i32> @var_shift_v4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v4i32':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 11 for instruction:   %shift
; AVX: Found an estimated cost of 11 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <4 x i32> %a, %b
  ret <4 x i32> %shift
}

define <8 x i32> @var_shift_v8i32(<8 x i32> %a, <8 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v8i32':
; SSE2: Found an estimated cost of 32 for instruction:   %shift
; SSE41: Found an estimated cost of 22 for instruction:   %shift
; AVX: Found an estimated cost of 22 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <8 x i32> %a, %b
  ret <8 x i32> %shift
}

define <16 x i32> @var_shift_v16i32(<16 x i32> %a, <16 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v16i32':
; SSE2: Found an estimated cost of 64 for instruction:   %shift
; SSE41: Found an estimated cost of 44 for instruction:   %shift
; AVX: Found an estimated cost of 44 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <16 x i32> %a, %b
  ret <16 x i32> %shift
}

define <8 x i16> @var_shift_v8i16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v8i16':
; SSE2: Found an estimated cost of 32 for instruction:   %shift
; SSE41: Found an estimated cost of 14 for instruction:   %shift
; AVX: Found an estimated cost of 14 for instruction:   %shift
; AVX2: Found an estimated cost of 14 for instruction:   %shift
; AVX512: Found an estimated cost of 14 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <8 x i16> %a, %b
  ret <8 x i16> %shift
}

define <16 x i16> @var_shift_v16i16(<16 x i16> %a, <16 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v16i16':
; SSE2: Found an estimated cost of 64 for instruction:   %shift
; SSE41: Found an estimated cost of 28 for instruction:   %shift
; AVX: Found an estimated cost of 28 for instruction:   %shift
; AVX2: Found an estimated cost of 10 for instruction:   %shift
; AVX512: Found an estimated cost of 10 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %shift = lshr <16 x i16> %a, %b
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
; AVX512VL: Found an estimated cost of 20 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 1 for instruction:   %shift
; XOP: Found an estimated cost of 8 for instruction:   %shift
  %shift = lshr <32 x i16> %a, %b
  ret <32 x i16> %shift
}

define <16 x i8> @var_shift_v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v16i8':
; SSE2: Found an estimated cost of 26 for instruction:   %shift
; SSE41: Found an estimated cost of 12 for instruction:   %shift
; AVX: Found an estimated cost of 12 for instruction:   %shift
; AVX2: Found an estimated cost of 12 for instruction:   %shift
; AVX512: Found an estimated cost of 12 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <16 x i8> %a, %b
  ret <16 x i8> %shift
}

define <32 x i8> @var_shift_v32i8(<32 x i8> %a, <32 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v32i8':
; SSE2: Found an estimated cost of 52 for instruction:   %shift
; SSE41: Found an estimated cost of 24 for instruction:   %shift
; AVX: Found an estimated cost of 24 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %shift = lshr <32 x i8> %a, %b
  ret <32 x i8> %shift
}

define <64 x i8> @var_shift_v64i8(<64 x i8> %a, <64 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'var_shift_v64i8':
; SSE2: Found an estimated cost of 104 for instruction:   %shift
; SSE41: Found an estimated cost of 48 for instruction:   %shift
; AVX: Found an estimated cost of 48 for instruction:   %shift
; AVX2: Found an estimated cost of 22 for instruction:   %shift
; AVX512F: Found an estimated cost of 22 for instruction:   %shift
; AVX512BW: Found an estimated cost of 11 for instruction:   %shift
; AVX512VL: Found an estimated cost of 22 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 8 for instruction:   %shift
  %shift = lshr <64 x i8> %a, %b
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
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <2 x i64> %b, <2 x i64> undef, <2 x i32> zeroinitializer
  %shift = lshr <2 x i64> %a, %splat
  ret <2 x i64> %shift
}

define <4 x i64> @splatvar_shift_v4i64(<4 x i64> %a, <4 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v4i64':
; SSE2: Found an estimated cost of 8 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <4 x i64> %b, <4 x i64> undef, <4 x i32> zeroinitializer
  %shift = lshr <4 x i64> %a, %splat
  ret <4 x i64> %shift
}

define <8 x i64> @splatvar_shift_v8i64(<8 x i64> %a, <8 x i64> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v8i64':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 16 for instruction:   %shift
; AVX: Found an estimated cost of 16 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %splat = shufflevector <8 x i64> %b, <8 x i64> undef, <8 x i32> zeroinitializer
  %shift = lshr <8 x i64> %a, %splat
  ret <8 x i64> %shift
}

define <4 x i32> @splatvar_shift_v4i32(<4 x i32> %a, <4 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v4i32':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 11 for instruction:   %shift
; AVX: Found an estimated cost of 11 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <4 x i32> %b, <4 x i32> undef, <4 x i32> zeroinitializer
  %shift = lshr <4 x i32> %a, %splat
  ret <4 x i32> %shift
}

define <8 x i32> @splatvar_shift_v8i32(<8 x i32> %a, <8 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v8i32':
; SSE2: Found an estimated cost of 32 for instruction:   %shift
; SSE41: Found an estimated cost of 22 for instruction:   %shift
; AVX: Found an estimated cost of 22 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %splat = shufflevector <8 x i32> %b, <8 x i32> undef, <8 x i32> zeroinitializer
  %shift = lshr <8 x i32> %a, %splat
  ret <8 x i32> %shift
}

define <16 x i32> @splatvar_shift_v16i32(<16 x i32> %a, <16 x i32> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v16i32':
; SSE2: Found an estimated cost of 64 for instruction:   %shift
; SSE41: Found an estimated cost of 44 for instruction:   %shift
; AVX: Found an estimated cost of 44 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %splat = shufflevector <16 x i32> %b, <16 x i32> undef, <16 x i32> zeroinitializer
  %shift = lshr <16 x i32> %a, %splat
  ret <16 x i32> %shift
}

define <8 x i16> @splatvar_shift_v8i16(<8 x i16> %a, <8 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v8i16':
; SSE2: Found an estimated cost of 32 for instruction:   %shift
; SSE41: Found an estimated cost of 14 for instruction:   %shift
; AVX: Found an estimated cost of 14 for instruction:   %shift
; AVX2: Found an estimated cost of 14 for instruction:   %shift
; AVX512: Found an estimated cost of 14 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %splat = shufflevector <8 x i16> %b, <8 x i16> undef, <8 x i32> zeroinitializer
  %shift = lshr <8 x i16> %a, %splat
  ret <8 x i16> %shift
}

define <16 x i16> @splatvar_shift_v16i16(<16 x i16> %a, <16 x i16> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v16i16':
; SSE2: Found an estimated cost of 64 for instruction:   %shift
; SSE41: Found an estimated cost of 28 for instruction:   %shift
; AVX: Found an estimated cost of 28 for instruction:   %shift
; AVX2: Found an estimated cost of 10 for instruction:   %shift
; AVX512: Found an estimated cost of 10 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %splat = shufflevector <16 x i16> %b, <16 x i16> undef, <16 x i32> zeroinitializer
  %shift = lshr <16 x i16> %a, %splat
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
; XOP: Found an estimated cost of 8 for instruction:   %shift
  %splat = shufflevector <32 x i16> %b, <32 x i16> undef, <32 x i32> zeroinitializer
  %shift = lshr <32 x i16> %a, %splat
  ret <32 x i16> %shift
}

define <16 x i8> @splatvar_shift_v16i8(<16 x i8> %a, <16 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v16i8':
; SSE2: Found an estimated cost of 26 for instruction:   %shift
; SSE41: Found an estimated cost of 12 for instruction:   %shift
; AVX: Found an estimated cost of 12 for instruction:   %shift
; AVX2: Found an estimated cost of 12 for instruction:   %shift
; AVX512: Found an estimated cost of 12 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %splat = shufflevector <16 x i8> %b, <16 x i8> undef, <16 x i32> zeroinitializer
  %shift = lshr <16 x i8> %a, %splat
  ret <16 x i8> %shift
}

define <32 x i8> @splatvar_shift_v32i8(<32 x i8> %a, <32 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v32i8':
; SSE2: Found an estimated cost of 52 for instruction:   %shift
; SSE41: Found an estimated cost of 24 for instruction:   %shift
; AVX: Found an estimated cost of 24 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %splat = shufflevector <32 x i8> %b, <32 x i8> undef, <32 x i32> zeroinitializer
  %shift = lshr <32 x i8> %a, %splat
  ret <32 x i8> %shift
}

define <64 x i8> @splatvar_shift_v64i8(<64 x i8> %a, <64 x i8> %b) {
; CHECK: 'Cost Model Analysis' for function 'splatvar_shift_v64i8':
; SSE2: Found an estimated cost of 104 for instruction:   %shift
; SSE41: Found an estimated cost of 48 for instruction:   %shift
; AVX: Found an estimated cost of 48 for instruction:   %shift
; AVX2: Found an estimated cost of 22 for instruction:   %shift
; AVX512F: Found an estimated cost of 22 for instruction:   %shift
; AVX512BW: Found an estimated cost of 11 for instruction:   %shift
; AVX512VL: Found an estimated cost of 22 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 8 for instruction:   %shift
  %splat = shufflevector <64 x i8> %b, <64 x i8> undef, <64 x i32> zeroinitializer
  %shift = lshr <64 x i8> %a, %splat
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
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <2 x i64> %a, <i64 1, i64 7>
  ret <2 x i64> %shift
}

define <4 x i64> @constant_shift_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v4i64':
; SSE2: Found an estimated cost of 8 for instruction:   %shift
; SSE41: Found an estimated cost of 8 for instruction:   %shift
; AVX: Found an estimated cost of 8 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <4 x i64> %a, <i64 1, i64 7, i64 15, i64 31>
  ret <4 x i64> %shift
}

define <8 x i64> @constant_shift_v8i64(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v8i64':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 16 for instruction:   %shift
; AVX: Found an estimated cost of 16 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <8 x i64> %a, <i64 1, i64 7, i64 15, i64 31, i64 1, i64 7, i64 15, i64 31>
  ret <8 x i64> %shift
}

define <4 x i32> @constant_shift_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v4i32':
; SSE2: Found an estimated cost of 16 for instruction:   %shift
; SSE41: Found an estimated cost of 11 for instruction:   %shift
; AVX: Found an estimated cost of 11 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 2 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <4 x i32> %a, <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i32> %shift
}

define <8 x i32> @constant_shift_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v8i32':
; SSE2: Found an estimated cost of 32 for instruction:   %shift
; SSE41: Found an estimated cost of 22 for instruction:   %shift
; AVX: Found an estimated cost of 22 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <8 x i32> %a, <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <8 x i32> %shift
}

define <16 x i32> @constant_shift_v16i32(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v16i32':
; SSE2: Found an estimated cost of 64 for instruction:   %shift
; SSE41: Found an estimated cost of 44 for instruction:   %shift
; AVX: Found an estimated cost of 44 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <16 x i32> %a, <i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 0, i32 1, i32 2, i32 3>
  ret <16 x i32> %shift
}

define <8 x i16> @constant_shift_v8i16(<8 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v8i16':
; SSE2: Found an estimated cost of 32 for instruction:   %shift
; SSE41: Found an estimated cost of 14 for instruction:   %shift
; AVX: Found an estimated cost of 14 for instruction:   %shift
; AVX2: Found an estimated cost of 14 for instruction:   %shift
; AVX512: Found an estimated cost of 14 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <8 x i16> %a, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>
  ret <8 x i16> %shift
}

define <16 x i16> @constant_shift_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v16i16':
; SSE2: Found an estimated cost of 64 for instruction:   %shift
; SSE41: Found an estimated cost of 28 for instruction:   %shift
; AVX: Found an estimated cost of 28 for instruction:   %shift
; AVX2: Found an estimated cost of 10 for instruction:   %shift
; AVX512: Found an estimated cost of 10 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %shift = lshr <16 x i16> %a, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>
  ret <16 x i16> %shift
}

define <32 x i16> @constant_shift_v32i16(<32 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v32i16':
; SSE2: Found an estimated cost of 128 for instruction:   %shift
; SSE41: Found an estimated cost of 56 for instruction:   %shift
; AVX: Found an estimated cost of 56 for instruction:   %shift
; AVX2: Found an estimated cost of 20 for instruction:   %shift
; AVX512F: Found an estimated cost of 20 for instruction:   %shift
; AVX512BW: Found an estimated cost of 1 for instruction:   %shift
; AVX512VL: Found an estimated cost of 20 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 1 for instruction:   %shift
; XOP: Found an estimated cost of 8 for instruction:   %shift
  %shift = lshr <32 x i16> %a, <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>
  ret <32 x i16> %shift
}

define <16 x i8> @constant_shift_v16i8(<16 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v16i8':
; SSE2: Found an estimated cost of 26 for instruction:   %shift
; SSE41: Found an estimated cost of 12 for instruction:   %shift
; AVX: Found an estimated cost of 12 for instruction:   %shift
; AVX2: Found an estimated cost of 12 for instruction:   %shift
; AVX512: Found an estimated cost of 12 for instruction:   %shift
; XOP: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <16 x i8> %a, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>
  ret <16 x i8> %shift
}

define <32 x i8> @constant_shift_v32i8(<32 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v32i8':
; SSE2: Found an estimated cost of 52 for instruction:   %shift
; SSE41: Found an estimated cost of 24 for instruction:   %shift
; AVX: Found an estimated cost of 24 for instruction:   %shift
; AVX2: Found an estimated cost of 11 for instruction:   %shift
; AVX512: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 4 for instruction:   %shift
  %shift = lshr <32 x i8> %a, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>
  ret <32 x i8> %shift
}

define <64 x i8> @constant_shift_v64i8(<64 x i8> %a) {
; CHECK: 'Cost Model Analysis' for function 'constant_shift_v64i8':
; SSE2: Found an estimated cost of 104 for instruction:   %shift
; SSE41: Found an estimated cost of 48 for instruction:   %shift
; AVX: Found an estimated cost of 48 for instruction:   %shift
; AVX2: Found an estimated cost of 22 for instruction:   %shift
; AVX512F: Found an estimated cost of 22 for instruction:   %shift
; AVX512BW: Found an estimated cost of 11 for instruction:   %shift
; AVX512VL: Found an estimated cost of 22 for instruction:   %shift
; AVX512BWVL: Found an estimated cost of 11 for instruction:   %shift
; XOP: Found an estimated cost of 8 for instruction:   %shift
  %shift = lshr <64 x i8> %a, <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 7, i8 6, i8 5, i8 4, i8 3, i8 2, i8 1, i8 0>
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
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <2 x i64> %a, <i64 7, i64 7>
  ret <2 x i64> %shift
}

define <4 x i64> @splatconstant_shift_v4i64(<4 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v4i64':
; SSE2: Found an estimated cost of 2 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 2 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <4 x i64> %a, <i64 7, i64 7, i64 7, i64 7>
  ret <4 x i64> %shift
}

define <8 x i64> @splatconstant_shift_v8i64(<8 x i64> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v8i64':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <8 x i64> %a, <i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7, i64 7>
  ret <8 x i64> %shift
}

define <4 x i32> @splatconstant_shift_v4i32(<4 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v4i32':
; SSE2: Found an estimated cost of 1 for instruction:   %shift
; SSE41: Found an estimated cost of 1 for instruction:   %shift
; AVX: Found an estimated cost of 1 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOP: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <4 x i32> %a, <i32 5, i32 5, i32 5, i32 5>
  ret <4 x i32> %shift
}

define <8 x i32> @splatconstant_shift_v8i32(<8 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v8i32':
; SSE2: Found an estimated cost of 2 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 2 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <8 x i32> %a, <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
  ret <8 x i32> %shift
}

define <16 x i32> @splatconstant_shift_v16i32(<16 x i32> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v16i32':
; SSE2: Found an estimated cost of 4 for instruction:   %shift
; SSE41: Found an estimated cost of 4 for instruction:   %shift
; AVX: Found an estimated cost of 4 for instruction:   %shift
; AVX2: Found an estimated cost of 2 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <16 x i32> %a, <i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5, i32 5>
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
  %shift = lshr <8 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
  ret <8 x i16> %shift
}

define <16 x i16> @splatconstant_shift_v16i16(<16 x i16> %a) {
; CHECK: 'Cost Model Analysis' for function 'splatconstant_shift_v16i16':
; SSE2: Found an estimated cost of 2 for instruction:   %shift
; SSE41: Found an estimated cost of 2 for instruction:   %shift
; AVX: Found an estimated cost of 2 for instruction:   %shift
; AVX2: Found an estimated cost of 1 for instruction:   %shift
; AVX512: Found an estimated cost of 1 for instruction:   %shift
; XOPAVX: Found an estimated cost of 4 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 1 for instruction:   %shift
  %shift = lshr <16 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
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
; XOPAVX: Found an estimated cost of 8 for instruction:   %shift
; XOPAVX2: Found an estimated cost of 2 for instruction:   %shift
  %shift = lshr <32 x i16> %a, <i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3, i16 3>
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
  %shift = lshr <16 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
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
  %shift = lshr <32 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
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
  %shift = lshr <64 x i8> %a, <i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3, i8 3>
  ret <64 x i8> %shift
}
