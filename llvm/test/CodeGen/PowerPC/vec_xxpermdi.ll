; RUN: llc -verify-machineinstrs -mtriple=powerpc64le-unknown-linux-gnu -mcpu=pwr8 < %s | \
; RUN:   FileCheck %s  -check-prefix=CHECK-LE
; RUN: llc -verify-machineinstrs -mtriple=powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | \
; RUN:   FileCheck %s -check-prefix=CHECK-BE

; Possible LE ShuffleVector masks (Case 1):
; ShuffleVector((vector double)a, (vector double)b, 3, 1)
; ShuffleVector((vector double)a, (vector double)b, 2, 1)
; ShuffleVector((vector double)a, (vector double)b, 3, 0)
; ShuffleVector((vector double)a, (vector double)b, 2, 0)
; which targets at:
; xxpermdi a, b, 0
; xxpermdi a, b, 1
; xxpermdi a, b, 2
; xxpermdi a, b, 3
; Possible LE Swap ShuffleVector masks (Case 2):
; ShuffleVector((vector double)a, (vector double)b, 1, 3)
; ShuffleVector((vector double)a, (vector double)b, 0, 3)
; ShuffleVector((vector double)a, (vector double)b, 1, 2)
; ShuffleVector((vector double)a, (vector double)b, 0, 2)
; which targets at:
; xxpermdi b, a, 0
; xxpermdi b, a, 1
; xxpermdi b, a, 2
; xxpermdi b, a, 3
; Possible LE ShuffleVector masks when a == b, b is undef (Case 3):
; ShuffleVector((vector double)a, (vector double)a, 1, 1)
; ShuffleVector((vector double)a, (vector double)a, 0, 1)
; ShuffleVector((vector double)a, (vector double)a, 1, 0)
; ShuffleVector((vector double)a, (vector double)a, 0, 0)
; which targets at:
; xxpermdi a, a, 0
; xxpermdi a, a, 1
; xxpermdi a, a, 2
; xxpermdi a, a, 3

; Possible BE ShuffleVector masks (Case 4):
; ShuffleVector((vector double)a, (vector double)b, 0, 2)
; ShuffleVector((vector double)a, (vector double)b, 0, 3)
; ShuffleVector((vector double)a, (vector double)b, 1, 2)
; ShuffleVector((vector double)a, (vector double)b, 1, 3)
; which targets at:
; xxpermdi a, b, 0
; xxpermdi a, b, 1
; xxpermdi a, b, 2
; xxpermdi a, b, 3
; Possible BE Swap ShuffleVector masks (Case 5):
; ShuffleVector((vector double)a, (vector double)b, 2, 0)
; ShuffleVector((vector double)a, (vector double)b, 3, 0)
; ShuffleVector((vector double)a, (vector double)b, 2, 1)
; ShuffleVector((vector double)a, (vector double)b, 3, 1)
; which targets at:
; xxpermdi b, a, 0
; xxpermdi b, a, 1
; xxpermdi b, a, 2
; xxpermdi b, a, 3
; Possible BE ShuffleVector masks when a == b, b is undef (Case 6):
; ShuffleVector((vector double)a, (vector double)a, 0, 0)
; ShuffleVector((vector double)a, (vector double)a, 0, 1)
; ShuffleVector((vector double)a, (vector double)a, 1, 0)
; ShuffleVector((vector double)a, (vector double)a, 1, 1)
; which targets at:
; xxpermdi a, a, 0
; xxpermdi a, a, 1
; xxpermdi a, a, 2
; xxpermdi a, a, 3

define <2 x double> @test_le_vec_xxpermdi_v2f64_v2f64_0(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 3, i32 1>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_vec_xxpermdi_v2f64_v2f64_0
; CHECK-LE: xxmrghd 34, 34, 35
; CHECK-LE: blr
}

define <2 x double> @test_le_vec_xxpermdi_v2f64_v2f64_1(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 2, i32 1>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_vec_xxpermdi_v2f64_v2f64_1
; CHECK-LE: xxpermdi 34, 34, 35, 1
; CHECK-LE: blr
}

define <2 x double> @test_le_vec_xxpermdi_v2f64_v2f64_2(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 3, i32 0>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_vec_xxpermdi_v2f64_v2f64_2
; CHECK-LE: xxpermdi 34, 34, 35, 2
; CHECK-LE: blr
}

define <2 x double> @test_le_vec_xxpermdi_v2f64_v2f64_3(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 2, i32 0>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_vec_xxpermdi_v2f64_v2f64_3
; CHECK-LE: xxmrgld 34, 34, 35
; CHECK-LE: blr
}

define <2 x double> @test_le_swap_vec_xxpermdi_v2f64_v2f64_0(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 1, i32 3>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_swap_vec_xxpermdi_v2f64_v2f64_0
; CHECK-LE: xxmrghd 34, 35, 34
; CHECK-LE: blr
}

define <2 x double> @test_le_swap_vec_xxpermdi_v2f64_v2f64_1(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 0, i32 3>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_swap_vec_xxpermdi_v2f64_v2f64_1
; CHECK-LE: xxpermdi 34, 35, 34, 1
; CHECK-LE: blr
}

define <2 x double> @test_le_swap_vec_xxpermdi_v2f64_v2f64_2(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 1, i32 2>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_swap_vec_xxpermdi_v2f64_v2f64_2
; CHECK-LE: xxpermdi 34, 35, 34, 2
; CHECK-LE: blr
}

define <2 x double> @test_le_swap_vec_xxpermdi_v2f64_v2f64_3(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 0, i32 2>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_swap_vec_xxpermdi_v2f64_v2f64_3
; CHECK-LE: xxmrgld 34, 35, 34
; CHECK-LE: blr
}

define <2 x double> @test_le_vec_xxpermdi_v2f64_undef_0(<2 x double> %VA) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> undef, <2 x i32> <i32 1, i32 1>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_vec_xxpermdi_v2f64_undef_0
; CHECK-LE: xxspltd 34, 34, 0
; CHECK-LE: blr
}

define <2 x double> @test_le_vec_xxpermdi_v2f64_undef_1(<2 x double> %VA) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> undef, <2 x i32> <i32 0, i32 1>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_vec_xxpermdi_v2f64_undef_1
; CHECK-LE: blr
}

define <2 x double> @test_le_vec_xxpermdi_v2f64_undef_2(<2 x double> %VA) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> undef, <2 x i32> <i32 1, i32 0>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_vec_xxpermdi_v2f64_undef_2
; CHECK-LE: xxswapd 34, 34
}

define <2 x double> @test_le_vec_xxpermdi_v2f64_undef_3(<2 x double> %VA) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> undef, <2 x i32> <i32 0, i32 0>
      ret <2 x double> %0
; CHECK-LE-LABEL: @test_le_vec_xxpermdi_v2f64_undef_3
; CHECK-LE: xxspltd 34, 34, 1
; CHECK-LE: blr
}

; Start testing BE
define <2 x double> @test_be_vec_xxpermdi_v2f64_v2f64_0(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 0, i32 2>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v2f64_v2f64_0
; CHECK-BE: xxmrghd 34, 34, 35
; CHECK-BE: blr
}

define <2 x double> @test_be_vec_xxpermdi_v2f64_v2f64_1(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 0, i32 3>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v2f64_v2f64_1
; CHECK-BE: xxpermdi 34, 34, 35, 1
; CHECK-BE: blr
}

define <2 x double> @test_be_vec_xxpermdi_v2f64_v2f64_2(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 1, i32 2>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v2f64_v2f64_2
; CHECK-BE: xxpermdi 34, 34, 35, 2
; CHECK-BE: blr
}

define <2 x double> @test_be_vec_xxpermdi_v2f64_v2f64_3(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 1, i32 3>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v2f64_v2f64_3
; CHECK-BE: xxmrgld 34, 34, 35
; CHECK-BE: blr
}

define <2 x double> @test_be_swap_vec_xxpermdi_v2f64_v2f64_0(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 2, i32 0>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_swap_vec_xxpermdi_v2f64_v2f64_0
; CHECK-BE: xxmrghd 34, 35, 34
; CHECK-BE: blr
}

define <2 x double> @test_be_swap_vec_xxpermdi_v2f64_v2f64_1(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 2, i32 1>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_swap_vec_xxpermdi_v2f64_v2f64_1
; CHECK-BE: xxpermdi 34, 35, 34, 1
; CHECK-BE: blr
}

define <2 x double> @test_be_swap_vec_xxpermdi_v2f64_v2f64_2(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 3, i32 0>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_swap_vec_xxpermdi_v2f64_v2f64_2
; CHECK-BE: xxpermdi 34, 35, 34, 2
; CHECK-BE: blr
}

define <2 x double> @test_be_swap_vec_xxpermdi_v2f64_v2f64_3(<2 x double> %VA, <2 x double> %VB) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> %VB,<2 x i32> <i32 3, i32 1>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_swap_vec_xxpermdi_v2f64_v2f64_3
; CHECK-BE: xxmrgld 34, 35, 34
; CHECK-BE: blr
}

define <2 x double> @test_be_vec_xxpermdi_v2f64_undef_0(<2 x double> %VA) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> undef, <2 x i32> <i32 0, i32 0>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v2f64_undef_0
; CHECK-BE: xxspltd 34, 34, 0
; CHECK-BE: blr
}

define <2 x double> @test_be_vec_xxpermdi_v2f64_undef_1(<2 x double> %VA) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> undef, <2 x i32> <i32 0, i32 1>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v2f64_undef_1
; CHECK-BE: blr
}

define <2 x double> @test_be_vec_xxpermdi_v2f64_undef_2(<2 x double> %VA) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> undef, <2 x i32> <i32 1, i32 0>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v2f64_undef_2
; CHECK-LE: xxswapd 34, 34
}

define <2 x double> @test_be_vec_xxpermdi_v2f64_undef_3(<2 x double> %VA) {
     entry:
      %0 = shufflevector <2 x double> %VA, <2 x double> undef, <2 x i32> <i32 1, i32 1>
      ret <2 x double> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v2f64_undef_3
; CHECK-BE: xxspltd 34, 34, 1
; CHECK-BE: blr
}

; More test cases to test different types of vector inputs
define <16 x i8> @test_be_vec_xxpermdi_v16i8_v16i8(<16 x i8> %VA, <16 x i8> %VB) {
     entry:
      %0 = shufflevector <16 x i8> %VA, <16 x i8> %VB,<16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19>
      ret <16 x i8> %0
; CHECK-BE-LABEL: @test_be_vec_xxpermdi_v16i8_v16i8
; CHECK-BE: xxpermdi 34, 34, 35, 1
; CHECK-BE: blr
}

define <8 x i16> @test_le_swap_vec_xxpermdi_v8i16_v8i16(<8 x i16> %VA, <8 x i16> %VB) {
     entry:
      %0 = shufflevector <8 x i16> %VA, <8 x i16> %VB,<8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 12, i32 13, i32 14, i32 15>
      ret <8 x i16> %0
; CHECK-LE-LABEL: @test_le_swap_vec_xxpermdi_v8i16_v8i16
; CHECK-LE: xxpermdi 34, 35, 34, 1
; CHECK-LE: blr
}

define <4 x i32> @test_le_swap_vec_xxpermdi_v4i32_v4i32(<4 x i32> %VA, <4 x i32> %VB) {
     entry:
      %0 = shufflevector <4 x i32> %VA, <4 x i32> %VB,<4 x i32> <i32 0, i32 1, i32 6, i32 7>
      ret <4 x i32> %0
; CHECK-LE-LABEL: @test_le_swap_vec_xxpermdi_v4i32_v4i32
; CHECK-LE: xxpermdi 34, 35, 34, 1
; CHECK-LE: blr
}
