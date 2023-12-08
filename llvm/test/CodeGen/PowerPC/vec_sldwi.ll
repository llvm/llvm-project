; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | \
; RUN:   FileCheck %s  -check-prefix=CHECK-LE
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | \
; RUN:   FileCheck %s -check-prefix=CHECK-BE

; Possible LE ShuffleVector masks (Case 1):
; ShuffleVector((vector int)a, vector(int)b, 0, 1, 2, 3)
; ShuffleVector((vector int)a, vector(int)b, 7, 0, 1, 2)
; ShuffleVector((vector int)a, vector(int)b, 6, 7, 0, 1)
; ShuffleVector((vector int)a, vector(int)b, 5, 6, 7, 0)
; which targets at:
; xxsldwi a, b, 0
; xxsldwi a, b, 1
; xxsldwi a, b, 2
; xxsldwi a, b, 3
; Possible LE Swap ShuffleVector masks (Case 2):
; ShuffleVector((vector int)a, vector(int)b, 4, 5, 6, 7)
; ShuffleVector((vector int)a, vector(int)b, 3, 4, 5, 6)
; ShuffleVector((vector int)a, vector(int)b, 2, 3, 4, 5)
; ShuffleVector((vector int)a, vector(int)b, 1, 2, 3, 4)
; which targets at:
; xxsldwi b, a, 0
; xxsldwi b, a, 1
; xxsldwi b, a, 2
; xxsldwi b, a, 3
; Possible LE ShuffleVector masks when a == b, b is undef (Case 3):
; ShuffleVector((vector int)a, vector(int)a, 0, 1, 2, 3)
; ShuffleVector((vector int)a, vector(int)a, 3, 0, 1, 2)
; ShuffleVector((vector int)a, vector(int)a, 2, 3, 0, 1)
; ShuffleVector((vector int)a, vector(int)a, 1, 2, 3, 0)
; which targets at:
; xxsldwi a, a, 0
; xxsldwi a, a, 1
; xxsldwi a, a, 2
; xxsldwi a, a, 3

; Possible BE ShuffleVector masks (Case 4):
; ShuffleVector((vector int)a, vector(int)b, 0, 1, 2, 3)
; ShuffleVector((vector int)a, vector(int)b, 1, 2, 3, 4)
; ShuffleVector((vector int)a, vector(int)b, 2, 3, 4, 5)
; ShuffleVector((vector int)a, vector(int)b, 3, 4, 5, 6)
; which targets at:
; xxsldwi b, a, 0
; xxsldwi b, a, 1
; xxsldwi a, a, 2
; xxsldwi a, a, 3
; Possible BE Swap ShuffleVector masks (Case 5):
; ShuffleVector((vector int)a, vector(int)b, 4, 5, 6, 7)
; ShuffleVector((vector int)a, vector(int)b, 5, 6, 7, 0)
; ShuffleVector((vector int)a, vector(int)b, 6, 7, 0, 1)
; ShuffleVector((vector int)a, vector(int)b, 7, 0, 1, 2)
; which targets at:
; xxsldwi b, a, 0
; xxsldwi b, a, 1
; xxsldwi b, a, 2
; xxsldwi b, a, 3
; Possible BE ShuffleVector masks when a == b, b is undef (Case 6):
; ShuffleVector((vector int)a, vector(int)b, 0, 1, 2, 3)
; ShuffleVector((vector int)a, vector(int)a, 1, 2, 3, 0)
; ShuffleVector((vector int)a, vector(int)a, 2, 3, 0, 1)
; ShuffleVector((vector int)a, vector(int)a, 3, 0, 1, 2)
; which targets at:
; xxsldwi a, a, 0
; xxsldwi a, a, 1
; xxsldwi a, a, 2
; xxsldwi a, a, 3

define <4 x i32> @check_le_vec_sldwi_va_vb_0(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_vec_sldwi_va_vb_0
; CHECK-LE: blr
}

define <4 x i32> @check_le_vec_sldwi_va_vb_1(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 7, i32 0, i32 1, i32 2>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_vec_sldwi_va_vb_1
; CHECK-LE: xxsldwi 34, 34, 35, 1
; CHECK-LE: blr
}

define <4 x i32> @check_le_vec_sldwi_va_vb_2(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_vec_sldwi_va_vb_2
; CHECK-LE: xxsldwi 34, 34, 35, 2
; CHECK-LE: blr
}

define <4 x i32> @check_le_vec_sldwi_va_vb_3(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 5, i32 6, i32 7, i32 0>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_vec_sldwi_va_vb_3
; CHECK-LE: xxsldwi 34, 34, 35, 3
; CHECK-LE: blr
}

define <4 x i32> @check_le_swap_vec_sldwi_va_vb_0(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_swap_vec_sldwi_va_vb_0
; CHECK-LE: vmr 2, 3
; CHECK-LE: blr
}

define <4 x i32> @check_le_swap_vec_sldwi_va_vb_1(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_swap_vec_sldwi_va_vb_1
; CHECK-LE: xxsldwi 34, 35, 34, 1
; CHECK-LE: blr
}

define <4 x i32> @check_le_swap_vec_sldwi_va_vb_2(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_swap_vec_sldwi_va_vb_2
; CHECK-LE: xxsldwi 34, 35, 34, 2
; CHECK-LE: blr
}

define <4 x i32> @check_le_swap_vec_sldwi_va_vb_3(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_swap_vec_sldwi_va_vb_3
; CHECK-LE: xxsldwi 34, 35, 34, 3
; CHECK-LE: blr
}

define <4 x i32> @check_le_vec_sldwi_va_undef_0(<4 x i32> %VA) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_vec_sldwi_va_undef_0
; CHECK-LE: blr
}

define <4 x i32> @check_le_vec_sldwi_va_undef_1(<4 x i32> %VA) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> undef, <4 x i32> <i32 3, i32 0, i32 1, i32 2>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_le_vec_sldwi_va_undef_1
; CHECK-LE: xxsldwi 34, 34, 34, 1
; CHECK-LE: blr
}

define <4 x i32> @check_le_vec_sldwi_va_undef_2(<4 x i32> %VA) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_vec_sldwi_va_undef_2
; CHECK-LE: xxswapd 34, 34
; CHECK-LE: blr
}

define <4 x i32> @check_le_vec_sldwi_va_undef_3(<4 x i32> %VA) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_le_vec_sldwi_va_undef_3
; CHECK-LE: xxsldwi 34, 34, 34, 3
; CHECK-LE: blr
}

define <4 x i32> @check_be_vec_sldwi_va_vb_0(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_vec_sldwi_va_vb_0
; CHECK-BE: blr
}

define <4 x i32> @check_be_vec_sldwi_va_vb_1(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 1, i32 2, i32 3, i32 4>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_vec_sldwi_va_vb_1
; CHECK-BE: xxsldwi 34, 34, 35, 1
; CHECK-BE: blr
}

define <4 x i32> @check_be_vec_sldwi_va_vb_2(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 2, i32 3, i32 4, i32 5>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_vec_sldwi_va_vb_2
; CHECK-BE: xxsldwi 34, 34, 35, 2
; CHECK-BE: blr
}

define <4 x i32> @check_be_vec_sldwi_va_vb_3(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 3, i32 4, i32 5, i32 6>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_vec_sldwi_va_vb_3
; CHECK-BE: xxsldwi 34, 34, 35, 3
; CHECK-BE: blr
}

define <4 x i32> @check_be_swap_vec_sldwi_va_vb_0(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 4, i32 5, i32 6, i32 7>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_swap_vec_sldwi_va_vb_0
; CHECK-LE: vmr 2, 3
; CHECK-BE: blr
}

define <4 x i32> @check_be_swap_vec_sldwi_va_vb_1(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 5, i32 6, i32 7, i32 0>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_swap_vec_sldwi_va_vb_1
; CHECK-BE: xxsldwi 34, 35, 34, 1
; CHECK-BE: blr
}

define <4 x i32> @check_be_swap_vec_sldwi_va_vb_2(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 6, i32 7, i32 0, i32 1>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_swap_vec_sldwi_va_vb_2
; CHECK-BE: xxsldwi 34, 35, 34, 2
; CHECK-BE: blr
}

define <4 x i32> @check_be_swap_vec_sldwi_va_vb_3(<4 x i32> %VA, <4 x i32> %VB) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB, <4 x i32> <i32 7, i32 0, i32 1, i32 2>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_swap_vec_sldwi_va_vb_3
; CHECK-BE: xxsldwi 34, 35, 34, 3
; CHECK-BE: blr
}

define <4 x i32> @check_be_vec_sldwi_va_undef_0(<4 x i32> %VA) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> undef, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %0
; CHECK-LE-LABEL: @check_be_vec_sldwi_va_undef_0
; CHECK-BE: blr
}

define <4 x i32> @check_be_vec_sldwi_va_undef_1(<4 x i32> %VA) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> undef, <4 x i32> <i32 1, i32 2, i32 3, i32 0>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_vec_sldwi_va_undef_1
; CHECK-BE: xxsldwi 34, 34, 34, 1
; CHECK-BE: blr
}

define <4 x i32> @check_be_vec_sldwi_va_undef_2(<4 x i32> %VA) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> undef, <4 x i32> <i32 2, i32 3, i32 0, i32 1>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_vec_sldwi_va_undef_2
; CHECK-BE: xxswapd 34, 34
; CHECK-BE: blr
}

define <4 x i32> @check_be_vec_sldwi_va_undef_3(<4 x i32> %VA) {
entry:
  %0 = shufflevector <4 x i32> %VA, <4 x i32> undef, <4 x i32> <i32 3, i32 0, i32 1, i32 2>
  ret <4 x i32> %0
; CHECK-BE-LABEL: @check_be_vec_sldwi_va_undef_3
; CHECK-BE: xxsldwi 34, 34, 34, 3
; CHECK-BE: blr
}

; More test cases to test different types of vector inputs
define <16 x i8> @test_le_vec_sldwi_v16i8_v16i8(<16 x i8> %VA, <16 x i8> %VB) {
     entry:
      %0 = shufflevector <16 x i8> %VA, <16 x i8> %VB,<16 x i32> <i32 28, i32 29, i32 30, i32 31,i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11>
      ret <16 x i8> %0
; CHECK-LE-LABEL: @test_le_vec_sldwi_v16i8_v16i8
; CHECK-LE: xxsldwi 34, 34, 35, 1
; CHECK-LE: blr
}

define <8 x i16> @test_le_vec_sldwi_v8i16_v8i16(<8 x i16> %VA, <8 x i16> %VB) {
     entry:
      %0 = shufflevector <8 x i16> %VA, <8 x i16> %VB,<8 x i32> <i32 14, i32 15, i32 0, i32 1, i32 2, i32 3, i32 4, i32 5>
      ret <8 x i16> %0
; CHECK-LE-LABEL: @test_le_vec_sldwi_v8i16_v8i16
; CHECK-LE: xxsldwi 34, 34, 35, 1
; CHECK-LE: blr
}

; Note here xxpermdi 34, 34, 35, 2 <=> xxsldwi 34, 34, 35, 2
define <2 x i64> @test_be_vec_sldwi_v2i64_v2i64(<2 x i64> %VA, <2 x i64> %VB) {
     entry:
      %0 = shufflevector <2 x i64> %VA, <2 x i64> %VB,<2 x i32> <i32 3, i32 0>
      ret <2 x i64> %0
; CHECK-LE-LABEL: @test_be_vec_sldwi_v2i64_v2i64
; CHECK-LE: xxpermdi 34, 34, 35, 2
; CHECK-LE: blr
}
