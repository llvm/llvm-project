; NOTE: Assertions have been simpilfied MANUALLY after running utils/update_llc_test_checks.py
;       Assertions for constant pools have been added MANUALLY.
; RUN: llc < %s -mtriple=i686-unknown -mattr=+avx | FileCheck %s --check-prefix=AVX
; RUN: llc < %s -mtriple=i686-unknown -mattr=+avx2 | FileCheck %s -check-prefix=ALL -check-prefix=ALL32 -check-prefix=NO-AVX512BW -check-prefix=AVX2 
; RUN: llc < %s -mtriple=i686-unknown -mattr=+avx512f | FileCheck %s -check-prefix=ALL -check-prefix=ALL32 -check-prefix=NO-AVX512BW -check-prefix=AVX512 
; RUN: llc < %s -mtriple=i686-unknown -mattr=+avx512f,+avx512bw | FileCheck %s -check-prefix=ALL -check-prefix=ALL32 -check-prefix=AVX512BW -check-prefix=AVX512 
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx | FileCheck %s --check-prefix=AVX
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx2 | FileCheck %s -check-prefix=ALL -check-prefix=ALL64 -check-prefix=NO-AVX512BW -check-prefix=AVX2 -check-prefix=AVX2-64
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx512f | FileCheck %s -check-prefix=ALL -check-prefix=ALL64 -check-prefix=NO-AVX512BW -check-prefix=AVX512 -check-prefix=AVX512F-64
; RUN: llc < %s -mtriple=x86_64-unknown -mattr=+avx512f,+avx512bw | FileCheck %s -check-prefix=ALL -check-prefix=ALL64 -check-prefix=AVX512BW -check-prefix=AVX512 -check-prefix=AVX512BW-64

;===-----------------------------------------------------------------------------===
;    This test checks the ability to recognize a cross element pattern of
;    constants and perform the load via broadcasting a smaller constant
;    vector.
;    For example:
;    <i32 0, i32 1, i32 0, i32 1> => broadcast of the constant vector <i32 0, i32 1>
;===-----------------------------------------------------------------------------===

; ALL:       LCPI0
; ALL-NEXT:  .short	256                     # 0x100

define <16 x i8> @f16xi8_i16(<16 x i8> %a) {
; ALL32-LABEL: f16xi8_i16:
; ALL32:       # BB#0:
; ALL32-NEXT:    vpbroadcastw {{\.LCPI.*}}, %xmm1
; ALL32-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f16xi8_i16:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastw {{.*}}(%rip), %xmm1
; ALL64-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    retq
  %res1 = add <16 x i8> <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>, %a
  %res2 = and <16 x i8> <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>, %res1
  ret <16 x i8> %res2
}


; ALL:       .LCPI1
; ALL-NEXT:  .long	50462976                # 0x3020100

; AVX:       .LCPI1
; AVX-NEXT   .long	50462976                # float 3.82047143E-37

define <16 x i8> @f16xi8_i32(<16 x i8> %a) {
; ALL32-LABEL: f16xi8_i32:
; ALL32:       # BB#0:
; ALL32-NEXT:    vpbroadcastd {{\.LCPI.*}}, %xmm1
; ALL32-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f16xi8_i32:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; ALL64-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f16xi8_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastss {{\.LCPI.*}}, %xmm1
; AVX-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
  %res1 = add <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>, %a
  %res2 = and <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>, %res1
  ret <16 x i8> %res2
}


; ALL64:       .LCPI2
; ALL64-NEXT:  .quad	506097522914230528      # 0x706050403020100

; AVX:         .LCPI2
; AVX-NEXT:    .quad	506097522914230528      # double 7.9499288951273625E-275

define <16 x i8> @f16xi8_i64(<16 x i8> %a) {
; ALL32-LABEL: f16xi8_i64:
; ALL32:       # BB#0:
; ALL32-NEXT:    vmovddup {{.*#+}} xmm1 = mem[0,0]
; ALL32-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f16xi8_i64:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastq {{.*}}(%rip), %xmm1
; ALL64-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f16xi8_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovddup {{.*#+}} xmm1 = mem[0,0]
; AVX-NEXT:    vpaddb %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
  %res1 = add <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>, %a
  %res2 = and <16 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>, %res1
  ret <16 x i8> %res2
}


; ALL:       .LCPI3
; ALL-NEXT:  .short	256                     # 0x100

define <32 x i8> @f32xi8_i16(<32 x i8> %a) {
; ALL32-LABEL: f32xi8_i16:
; ALL32:       # BB#0:
; ALL32-NEXT:    vpbroadcastw {{\.LCPI.*}}, %ymm1
; ALL32-NEXT:    vpaddb %ymm1, %ymm0, %ymm0
; ALL32-NEXT:    vpand %ymm1, %ymm0, %ymm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f32xi8_i16:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastw {{.*}}(%rip), %ymm1
; ALL64-NEXT:    vpaddb %ymm1, %ymm0, %ymm0
; ALL64-NEXT:    vpand %ymm1, %ymm0, %ymm0
; ALL64-NEXT:    retq
  %res1 = add <32 x i8> <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>, %a
  %res2 = and <32 x i8> <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>, %res1
  ret <32 x i8> %res2
}


; ALL:       .LCPI4
; ALL-NEXT:  .long	50462976                # 0x3020100

; AVX:       .LCPI4
; AVX-NEXT:  .long	50462976                # float 3.82047143E-37

define <32 x i8> @f32xi8_i32(<32 x i8> %a) {
; ALL32-LABEL: f32xi8_i32:
; ALL32:       # BB#0:
; ALL32-NEXT:    vpbroadcastd {{\.LCPI.*}}, %ymm1
; ALL32-NEXT:    vpaddb %ymm1, %ymm0, %ymm0
; ALL32-NEXT:    vpand %ymm1, %ymm0, %ymm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f32xi8_i32:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastd {{.*}}(%rip), %ymm1
; ALL64-NEXT:    vpaddb %ymm1, %ymm0, %ymm0
; ALL64-NEXT:    vpand %ymm1, %ymm0, %ymm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f32xi8_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NEXT:    vbroadcastss {{\.LCPI.*}}, %xmm2
; AVX-NEXT:    vpaddb %xmm2, %xmm1, %xmm1
; AVX-NEXT:    vpaddb %xmm2, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    vandps {{\.LCPI.*}}, %ymm0, %ymm0
  %res1 = add <32 x i8> <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>, %a
  %res2 = and <32 x i8> <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>, %res1
  ret <32 x i8> %res2
}


; ALL64:       .LCPI5
; ALL64-NEXT:  .quad	506097522914230528      # 0x706050403020100

; AVX:         .LCPI5
; AVX-NEXT:    .quad	506097522914230528      # double 7.9499288951273625E-275

define <32 x i8> @f32xi8_i64(<32 x i8> %a) {
; ALL32-LABEL: f32xi8_i64:
; ALL32:       # BB#0:
; ALL32-NEXT:    vpbroadcastq {{\.LCPI.*}}, %ymm1
; ALL32-NEXT:    vpaddb %ymm1, %ymm0, %ymm0
; ALL32-NEXT:    vpand %ymm1, %ymm0, %ymm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f32xi8_i64:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastq {{.*}}(%rip), %ymm1
; ALL64-NEXT:    vpaddb %ymm1, %ymm0, %ymm0
; ALL64-NEXT:    vpand %ymm1, %ymm0, %ymm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f32xi8_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NEXT:    vmovddup {{.*#+}} xmm2 = mem[0,0]
; AVX-NEXT:    vpaddb %xmm2, %xmm1, %xmm1
; AVX-NEXT:    vpaddb %xmm2, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    vandps {{\.LCPI.*}}, %ymm0, %ymm0
  %res1 = add <32 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>, %a
  %res2 = and <32 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>, %res1
  ret <32 x i8> %res2
}


; ALL:       .LCPI6
; ALL-NEXT:  .byte	0                       # 0x0
; ALL-NEXT:  .byte	1                       # 0x1
; ALL-NEXT:  .byte	2                       # 0x2
; ALL-NEXT:  .byte	3                       # 0x3
; ALL-NEXT:  .byte	4                       # 0x4
; ALL-NEXT:  .byte	5                       # 0x5
; ALL-NEXT:  .byte	6                       # 0x6
; ALL-NEXT:  .byte	7                       # 0x7
; ALL-NEXT:  .byte	8                       # 0x8
; ALL-NEXT:  .byte	9                       # 0x9
; ALL-NEXT:  .byte	10                      # 0xa
; ALL-NEXT:  .byte	11                      # 0xb
; ALL-NEXT:  .byte	12                      # 0xc
; ALL-NEXT:  .byte	13                      # 0xd
; ALL-NEXT:  .byte	14                      # 0xe
; ALL-NEXT:  .byte	15                      # 0xf
; ALL-NOT:   .byte

define <32 x i8> @f32xi8_i128(<32 x i8> %a) {
; ALL-LABEL: f32xi8_i128:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcasti128 {{.*#+}} ymm1 = mem[0,1,0,1]
; ALL-NEXT:    vpaddb %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vpand %ymm1, %ymm0, %ymm0
  %res1 = add <32 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, %a
  %res2 = and <32 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, %res1
  ret <32 x i8> %res2
}


; ALL:       .LCPI7
; ALL-NEXT:  .short	256                     # 0x100

define <64 x i8> @f64xi8_i16(<64 x i8> %a) {
; NO-AVX512BW-LABEL: f64xi8_i16:
; NO-AVX512BW:       # BB#0:
; NO-AVX512BW-NEXT:    vpbroadcastw {{\.LCPI.*}}, %ymm2
; NO-AVX512BW-NEXT:    vpaddb %ymm2, %ymm1, %ymm1
; NO-AVX512BW-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512BW-LABEL: f64xi8_i16:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vpbroadcastw {{\.LCPI.*}}, %zmm1
; AVX512BW-NEXT:    vpaddb %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
  %res1 = add <64 x i8> <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>, %a
  %res2 = and <64 x i8> <i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1>, %res1
  ret <64 x i8> %res2
}


; ALL:       .LCPI8
; ALL-NEXT:  .long	50462976                # 0x3020100

; AVX:       .LCPI8
; AVX-NEXT:  .long	50462976                # float 3.82047143E-37

define <64 x i8> @f64i8_i32(<64 x i8> %a) {
; NO-AVX512BW-LABEL: f64i8_i32:
; NO-AVX512BW:       # BB#0:
; NO-AVX512BW-NEXT:    vpbroadcastd {{\.LCPI.*}}, %ymm2
; NO-AVX512BW-NEXT:    vpaddb %ymm2, %ymm1, %ymm1
; NO-AVX512BW-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512BW-LABEL: f64i8_i32:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vpbroadcastd {{\.LCPI.*}}, %zmm1
; AVX512BW-NEXT:    vpaddb %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
;
; AVX-LABEL: f64i8_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX-NEXT:    vbroadcastss {{\.LCPI.*}}, %xmm3
; AVX-NEXT:    vpaddb %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddb %xmm3, %xmm1, %xmm1
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX-NEXT:    vpaddb %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddb %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX-NEXT:    vmovaps {{.*#+}} ymm2 = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX-NEXT:    vandps %ymm2, %ymm1, %ymm1
  %res1 = add <64 x i8> <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>, %a
  %res2 = and <64 x i8> <i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3, i8 0, i8 1, i8 2, i8 3>, %res1
  ret <64 x i8> %res2
}


; ALL64:         .LCPI9
; ALL64-NEXT:    .quad	506097522914230528      # 0x706050403020100

; ALL32:         .LCPI9
; ALL32-NEXT:    .quad	506097522914230528      # double 7.9499288951273625E-275

; AVX:           .LCPI9
; AVX-NEXT:      .quad	506097522914230528      # double 7.9499288951273625E-275

define <64 x i8> @f64xi8_i64(<64 x i8> %a) {
; NO-AVX512BW-LABEL: f64xi8_i64:
; NO-AVX512BW:       # BB#0:
; NO-AVX512BW-NEXT:    vpbroadcastq {{.*}}, %ymm2
; NO-AVX512BW-NEXT:    vpaddb %ymm2, %ymm1, %ymm1
; NO-AVX512BW-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512BW-LABEL: f64xi8_i64:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vpbroadcastq {{.*}}, %zmm1
; AVX512BW-NEXT:    vpaddb %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
;
; AVX-LABEL: f64xi8_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX-NEXT:    vmovddup {{.*#+}} xmm3 = mem[0,0]
; AVX-NEXT:    vpaddb %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddb %xmm3, %xmm1, %xmm1
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX-NEXT:    vpaddb %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddb %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX-NEXT:    vmovaps {{.*#+}} ymm2 = [0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7,0,1,2,3,4,5,6,7]
; AVX-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX-NEXT:    vandps %ymm2, %ymm1, %ymm1
  %res1 = add <64 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>, %a
  %res2 = and <64 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7>, %res1
  ret <64 x i8> %res2
}


; ALL:       .LCPI10
; ALL-NEXT:  .byte	0                       # 0x0
; ALL-NEXT:  .byte	1                       # 0x1
; ALL-NEXT:  .byte	2                       # 0x2
; ALL-NEXT:  .byte	3                       # 0x3
; ALL-NEXT:  .byte	4                       # 0x4
; ALL-NEXT:  .byte	5                       # 0x5
; ALL-NEXT:  .byte	6                       # 0x6
; ALL-NEXT:  .byte	7                       # 0x7
; ALL-NEXT:  .byte	8                       # 0x8
; ALL-NEXT:  .byte	9                       # 0x9
; ALL-NEXT:  .byte	10                      # 0xa
; ALL-NEXT:  .byte	11                      # 0xb
; ALL-NEXT:  .byte	12                      # 0xc
; ALL-NEXT:  .byte	13                      # 0xd
; ALL-NEXT:  .byte	14                      # 0xe
; ALL-NEXT:  .byte	15                      # 0xf
; ALL-NOT:   .byte

define <64 x i8> @f64xi8_i128(<64 x i8> %a) {
; NO-AVX512BW-LABEL: f64xi8_i128:
; NO-AVX512BW:       # BB#0:
; NO-AVX512BW-NEXT:    vbroadcasti128 {{.*#+}} ymm2 = mem[0,1,0,1]
; NO-AVX512BW-NEXT:    vpaddb %ymm2, %ymm1, %ymm1
; NO-AVX512BW-NEXT:    vpaddb %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512BW-LABEL: f64xi8_i128:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vbroadcasti32x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX512BW-NEXT:    vpaddb %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
  %res1 = add <64 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, %a
  %res2 = and <64 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15>, %res1
  ret <64 x i8> %res2
}


; AVX512BW:       .LCPI11
; AVX512BW-NEXT:  .byte	0                       # 0x0
; AVX512BW-NEXT:  .byte	1                       # 0x1
; AVX512BW-NEXT:  .byte	2                       # 0x2
; AVX512BW-NEXT:  .byte	3                       # 0x3
; AVX512BW-NEXT:  .byte	4                       # 0x4
; AVX512BW-NEXT:  .byte	5                       # 0x5
; AVX512BW-NEXT:  .byte	6                       # 0x6
; AVX512BW-NEXT:  .byte	7                       # 0x7
; AVX512BW-NEXT:  .byte	8                       # 0x8
; AVX512BW-NEXT:  .byte	9                       # 0x9
; AVX512BW-NEXT:  .byte	10                      # 0xa
; AVX512BW-NEXT:  .byte	11                      # 0xb
; AVX512BW-NEXT:  .byte	12                      # 0xc
; AVX512BW-NEXT:  .byte	13                      # 0xd
; AVX512BW-NEXT:  .byte	14                      # 0xe
; AVX512BW-NEXT:  .byte	15                      # 0xf
; AVX512BW-NEXT:  .byte	16                      # 0x10
; AVX512BW-NEXT:  .byte	17                      # 0x11
; AVX512BW-NEXT:  .byte	18                      # 0x12
; AVX512BW-NEXT:  .byte	19                      # 0x13
; AVX512BW-NEXT:  .byte	20                      # 0x14
; AVX512BW-NEXT:  .byte	21                      # 0x15
; AVX512BW-NEXT:  .byte	22                      # 0x16
; AVX512BW-NEXT:  .byte	23                      # 0x17
; AVX512BW-NEXT:  .byte	24                      # 0x18
; AVX512BW-NEXT:  .byte	25                      # 0x19
; AVX512BW-NEXT:  .byte	26                      # 0x1a
; AVX512BW-NEXT:  .byte	27                      # 0x1b
; AVX512BW-NEXT:  .byte	28                      # 0x1c
; AVX512BW-NEXT:  .byte	29                      # 0x1d
; AVX512BW-NEXT:  .byte	30                      # 0x1e
; AVX512BW-NEXT:  .byte	31                      # 0x1f
; AVX512BW-NOT:   .byte

define <64 x i8> @f64xi8_i256(<64 x i8> %a) {
; AVX512BW-LABEL: f64xi8_i256:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vbroadcasti64x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3]
; AVX512BW-NEXT:    vpaddb %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
  %res1 = add <64 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, %a
  %res2 = and <64 x i8> <i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 10, i8 11, i8 12, i8 13, i8 14, i8 15, i8 16, i8 17, i8 18, i8 19, i8 20, i8 21, i8 22, i8 23, i8 24, i8 25, i8 26, i8 27, i8 28, i8 29, i8 30, i8 31>, %res1
  ret <64 x i8> %res2
}


; ALL:       .LCPI12
; ALL-NEXT:  .long	65536                   # 0x10000

; AVX:       .LCPI12
; AVX-NEXT:  .long	65536                   # float 9.18354962E-41

define <8 x i16> @f8xi16_i32(<8 x i16> %a) {
; ALL32-LABEL: f8xi16_i32:
; ALL32:       # BB#0:
; ALL32-NEXT:    vpbroadcastd {{\.LCPI.*}}, %xmm1
; ALL32-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f8xi16_i32:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; ALL64-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f8xi16_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastss {{\.LCPI.*}}, %xmm1
; AVX-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
  %res1 = add <8 x i16> <i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1>, %a
  %res2 = and <8 x i16> <i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1>, %res1
  ret <8 x i16> %res2
}


; ALL64:       .LCPI13
; ALL64-NEXT:  .quad	844433520132096         # 0x3000200010000

; ALL32:       .LCPI13
; ALL32-NEXT:  .quad	844433520132096         # double 4.1720559249406128E-309

; AVX:         .LCPI13
; AVX-NEXT:    .quad	844433520132096         # double 4.1720559249406128E-309

define <8 x i16> @f8xi16_i64(<8 x i16> %a) {
; ALL32-LABEL: f8xi16_i64:
; ALL32:       # BB#0:
; ALL32-NEXT:    vmovddup {{.*#+}} xmm1 = mem[0,0]
; ALL32-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f8xi16_i64:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastq {{.*}}(%rip), %xmm1
; ALL64-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f8xi16_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovddup {{.*#+}} xmm1 = mem[0,0]
; AVX-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
  %res1 = add <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3>, %a
  %res2 = and <8 x i16> <i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3>, %res1
  ret <8 x i16> %res2
}


; ALL:       .LCPI14
; ALL-NEXT:  .long	65536                   # 0x10000

; AVX:       .LCPI14
; AVX-NEXT:  .long	65536                   # float 9.18354962E-41

define <16 x i16> @f16xi16_i32(<16 x i16> %a) {
; ALL-LABEL: f16xi16_i32:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastd {{\.LCPI.*}}, %ymm1
; ALL-NEXT:    vpaddw %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vpand %ymm1, %ymm0, %ymm0
;
; AVX-LABEL: f16xi16_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NEXT:    vbroadcastss {{\.LCPI.*}}, %xmm2
; AVX-NEXT:    vpaddw %xmm2, %xmm1, %xmm1
; AVX-NEXT:    vpaddw %xmm2, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    vandps {{\.LCPI.*}}, %ymm0, %ymm0
  %res1 = add <16 x i16> <i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1>, %a
  %res2 = and <16 x i16> <i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1>, %res1
  ret <16 x i16> %res2
}


; ALL64:       .LCPI15
; ALL64-NEXT:  .quad	844433520132096         # 0x3000200010000

; ALL32:       .LCPI15
; ALL32-NEXT:  .quad	844433520132096         # double 4.1720559249406128E-309

; AVX:         .LCPI15
; AVX-NEXT:    .quad	844433520132096         # double 4.1720559249406128E-309

define <16 x i16> @f16xi16_i64(<16 x i16> %a) {
; ALL-LABEL: f16xi16_i64:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq {{.*}}, %ymm1
; ALL-NEXT:    vpaddw %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vpand %ymm1, %ymm0, %ymm0
;
; AVX-LABEL: f16xi16_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NEXT:    vmovddup {{.*#+}} xmm2 = mem[0,0]
; AVX-NEXT:    vpaddw %xmm2, %xmm1, %xmm1
; AVX-NEXT:    vpaddw %xmm2, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    vandps {{\.LCPI.*}}, %ymm0, %ymm0
  %res1 = add <16 x i16> <i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3>, %a
  %res2 = and <16 x i16> <i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3>, %res1
  ret <16 x i16> %res2
}


; ALL:       .LCPI16
; ALL-NEXT:  .short	0                       # 0x0
; ALL-NEXT:  .short	1                       # 0x1
; ALL-NEXT:  .short	2                       # 0x2
; ALL-NEXT:  .short	3                       # 0x3
; ALL-NEXT:  .short	4                       # 0x4
; ALL-NEXT:  .short	5                       # 0x5
; ALL-NEXT:  .short	6                       # 0x6
; ALL-NEXT:  .short	7                       # 0x7
; ALL-NOT:   .short

define <16 x i16> @f16xi16_i128(<16 x i16> %a) {
; ALL-LABEL: f16xi16_i128:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcasti128 {{.*#+}} ymm1 = mem[0,1,0,1]
; ALL-NEXT:    vpaddw %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vpand %ymm1, %ymm0, %ymm0
  %res1 = add <16 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %a
  %res2 = and <16 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %res1
  ret <16 x i16> %res2
}


; ALL:       .LCPI17
; ALL-NEXT:  .long	65536                   # 0x10000

; AVX:       .LCPI17
; AVX-NEXT:  .long	65536                   # float 9.18354962E-41

define <32 x i16> @f32xi16_i32(<32 x i16> %a) {
; NO-AVX512BW-LABEL: f32xi16_i32:
; NO-AVX512BW:       # BB#0:
; NO-AVX512BW-NEXT:    vpbroadcastd {{\.LCPI.*}}, %ymm2
; NO-AVX512BW-NEXT:    vpaddw %ymm2, %ymm1, %ymm1
; NO-AVX512BW-NEXT:    vpaddw %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512BW-LABEL: f32xi16_i32:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vpbroadcastd {{\.LCPI.*}}, %zmm1
; AVX512BW-NEXT:    vpaddw %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
;
; AVX-LABEL: f32xi16_i32:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX-NEXT:    vbroadcastss {{\.LCPI.*}}, %xmm3
; AVX-NEXT:    vpaddw %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddw %xmm3, %xmm1, %xmm1
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX-NEXT:    vpaddw %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddw %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX-NEXT:    vmovaps {{.*#+}} ymm2 = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
; AVX-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX-NEXT:    vandps %ymm2, %ymm1, %ymm1
  %res1 = add <32 x i16> <i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1>, %a
  %res2 = and <32 x i16> <i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1, i16 0, i16 1>, %res1
  ret <32 x i16> %res2
}


; ALL64:         .LCPI18
; ALL64-NEXT:    .quad	844433520132096         # 0x3000200010000

; ALL32:         .LCPI18
; ALL32-NEXT:    .quad	844433520132096         # double 4.1720559249406128E-309

; AVX:           .LCPI18
; AVX-NEXT:      .quad	844433520132096         # double 4.1720559249406128E-309

define <32 x i16> @f32xi16_i64(<32 x i16> %a) {
; NO-AVX512BW-LABEL: f32xi16_i64:
; NO-AVX512BW:       # BB#0:
; NO-AVX512BW-NEXT:    vpbroadcastq {{.*}}, %ymm2
; NO-AVX512BW-NEXT:    vpaddw %ymm2, %ymm1, %ymm1
; NO-AVX512BW-NEXT:    vpaddw %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512BW-LABEL: f32xi16_i64:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vpbroadcastq {{.*}}, %zmm1
; AVX512BW-NEXT:    vpaddw %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
;
; AVX-LABEL: f32xi16_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX-NEXT:    vmovddup {{.*#+}} xmm3 = mem[0,0]
; AVX-NEXT:    vpaddw %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddw %xmm3, %xmm1, %xmm1
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX-NEXT:    vpaddw %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddw %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX-NEXT:    vmovaps {{.*#+}} ymm2 = [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX-NEXT:    vandps %ymm2, %ymm1, %ymm1
  %res1 = add <32 x i16> <i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3>, %a
  %res2 = and <32 x i16> <i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3, i16 0, i16 1, i16 2, i16 3>, %res1
  ret <32 x i16> %res2
}


; ALL:       .LCPI19
; ALL-NEXT:  .short	0                       # 0x0
; ALL-NEXT:  .short	1                       # 0x1
; ALL-NEXT:  .short	2                       # 0x2
; ALL-NEXT:  .short	3                       # 0x3
; ALL-NEXT:  .short	4                       # 0x4
; ALL-NEXT:  .short	5                       # 0x5
; ALL-NEXT:  .short	6                       # 0x6
; ALL-NEXT:  .short	7                       # 0x7
; ALL-NOT:   .short

define <32 x i16> @f32xi16_i128(<32 x i16> %a) {
; NO-AVX512BW-LABEL: f32xi16_i128:
; NO-AVX512BW:       # BB#0:
; NO-AVX512BW-NEXT:    vbroadcasti128 {{.*#+}} ymm2 = mem[0,1,0,1]
; NO-AVX512BW-NEXT:    vpaddw %ymm2, %ymm1, %ymm1
; NO-AVX512BW-NEXT:    vpaddw %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm0, %ymm0
; NO-AVX512BW-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512BW-LABEL: f32xi16_i128:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vbroadcasti32x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX512BW-NEXT:    vpaddw %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
  %res1 = add <32 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %a
  %res2 = and <32 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7>, %res1
  ret <32 x i16> %res2
}


; AVX512BW:       .LCPI20
; AVX512BW-NEXT:  .short	0                       # 0x0
; AVX512BW-NEXT:  .short	1                       # 0x1
; AVX512BW-NEXT:  .short	2                       # 0x2
; AVX512BW-NEXT:  .short	3                       # 0x3
; AVX512BW-NEXT:  .short	4                       # 0x4
; AVX512BW-NEXT:  .short	5                       # 0x5
; AVX512BW-NEXT:  .short	6                       # 0x6
; AVX512BW-NEXT:  .short	7                       # 0x7
; AVX512BW-NEXT:  .short	8                       # 0x8
; AVX512BW-NEXT:  .short	9                       # 0x9
; AVX512BW-NEXT:  .short	10                      # 0xa
; AVX512BW-NEXT:  .short	11                      # 0xb
; AVX512BW-NEXT:  .short	12                      # 0xc
; AVX512BW-NEXT:  .short	13                      # 0xd
; AVX512BW-NEXT:  .short	14                      # 0xe
; AVX512BW-NEXT:  .short	15                      # 0xf
; AVX512BW-NOT:   .short

define <32 x i16> @f32xi16_i256(<32 x i16> %a) {
; AVX512BW-LABEL: f32xi16_i256:
; AVX512BW:       # BB#0:
; AVX512BW-NEXT:    vbroadcasti64x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3]
; AVX512BW-NEXT:    vpaddw %zmm1, %zmm0, %zmm0
; AVX512BW-NEXT:    vpandq %zmm1, %zmm0, %zmm0
  %res1 = add <32 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, %a
  %res2 = and <32 x i16> <i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15, i16 0, i16 1, i16 2, i16 3, i16 4, i16 5, i16 6, i16 7, i16 8, i16 9, i16 10, i16 11, i16 12, i16 13, i16 14, i16 15>, %res1
  ret <32 x i16> %res2
}


; ALL64:       .LCPI21
; ALL64-NEXT:  .quad	4294967296              # 0x100000000

; ALL32:       .LCPI21
; ALL32-NEXT:  .quad	4294967296              # double 2.1219957909652723E-314

; AVX:         .LCPI21
; AVX-NEXT:    .quad	4294967296              # double 2.1219957909652723E-314

define <4 x i32> @f4xi32_i64(<4 x i32> %a) {
; ALL32-LABEL: f4xi32_i64:
; ALL32:       # BB#0:
; ALL32-NEXT:    vmovddup {{.*#+}} xmm1 = mem[0,0]
; ALL32-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f4xi32_i64:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastq {{.*}}(%rip), %xmm1
; ALL64-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f4xi32_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovddup {{.*#+}} xmm1 = mem[0,0]
; AVX-NEXT:    vpaddd %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
  %res1 = add <4 x i32> <i32 0, i32 1, i32 0, i32 1>, %a
  %res2 = and <4 x i32> <i32 0, i32 1, i32 0, i32 1>, %res1
  ret <4 x i32> %res2
}


; ALL64:       .LCPI22
; ALL64-NEXT:  .quad	4294967296              # 0x100000000

; ALL32:       .LCPI22
; ALL32-NEXT:  .quad	4294967296              # double 2.1219957909652723E-314

; AVX:         .LCPI22
; AVX-NEXT:    .quad	4294967296              # double 2.1219957909652723E-314

define <8 x i32> @f8xi32_i64(<8 x i32> %a) {
; ALL-LABEL: f8xi32_i64:
; ALL:       # BB#0:
; ALL-NEXT:    vpbroadcastq {{.*}}, %ymm1
; ALL-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vpand %ymm1, %ymm0, %ymm0
;
; AVX-LABEL: f8xi32_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm1
; AVX-NEXT:    vmovddup {{.*#+}} xmm2 = mem[0,0]
; AVX-NEXT:    vpaddd %xmm2, %xmm1, %xmm1
; AVX-NEXT:    vpaddd %xmm2, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm1, %ymm0, %ymm0
; AVX-NEXT:    vandps {{\.LCPI.*}}, %ymm0, %ymm0
  %res1 = add <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>, %a
  %res2 = and <8 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>, %res1
  ret <8 x i32> %res2
}


; ALL:       .LCPI23
; ALL-NEXT:  .long	0                       # 0x0
; ALL-NEXT:  .long	1                       # 0x1
; ALL-NEXT:  .long	2                       # 0x2
; ALL-NEXT:  .long	3                       # 0x3
; ALL-NOT:   .long

define <8 x i32> @f8xi32_i128(<8 x i32> %a) {
; ALL-LABEL: f8xi32_i128:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcasti128 {{.*#+}} ymm1 = mem[0,1,0,1]
; ALL-NEXT:    vpaddd %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vpand %ymm1, %ymm0, %ymm0
  %res1 = add <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>, %a
  %res2 = and <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>, %res1
  ret <8 x i32> %res2
}


; ALL64:         .LCPI24
; ALL64-NEXT:    .quad	4294967296              # 0x100000000

; ALL32:         .LCPI24
; ALL32-NEXT:    .quad	4294967296              # double 2.1219957909652723E-314

; AVX:           .LCPI24
; AVX-NEXT:      .quad	4294967296              # double 2.1219957909652723E-314

define <16 x i32> @f16xi32_i64(<16 x i32> %a) {
; AVX2-LABEL: f16xi32_i64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vpbroadcastq {{.*}}, %ymm2
; AVX2-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpaddd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512-LABEL: f16xi32_i64:
; AVX512:       # BB#0:
; AVX512-NEXT:    vpbroadcastq {{.*}}, %zmm1
; AVX512-NEXT:    vpaddd %zmm1, %zmm0, %zmm0
; AVX512-NEXT:    vpandq %zmm1, %zmm0, %zmm0
;
; AVX-LABEL: f16xi32_i64:
; AVX:       # BB#0:
; AVX-NEXT:    vextractf128 $1, %ymm1, %xmm2
; AVX-NEXT:    vmovddup {{.*#+}} xmm3 = mem[0,0]
; AVX-NEXT:    vpaddd %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddd %xmm3, %xmm1, %xmm1
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm1, %ymm1
; AVX-NEXT:    vextractf128 $1, %ymm0, %xmm2
; AVX-NEXT:    vpaddd %xmm3, %xmm2, %xmm2
; AVX-NEXT:    vpaddd %xmm3, %xmm0, %xmm0
; AVX-NEXT:    vinsertf128 $1, %xmm2, %ymm0, %ymm0
; AVX-NEXT:    vmovaps {{.*#+}} ymm2 = [0,1,0,1,0,1,0,1]
; AVX-NEXT:    vandps %ymm2, %ymm0, %ymm0
; AVX-NEXT:    vandps %ymm2, %ymm1, %ymm1
  %res1 = add <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>, %a
  %res2 = and <16 x i32> <i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1, i32 0, i32 1>, %res1
  ret <16 x i32> %res2
}


; ALL:       .LCPI25
; ALL-NEXT:  .long	0                       # 0x0
; ALL-NEXT:  .long	1                       # 0x1
; ALL-NEXT:  .long	2                       # 0x2
; ALL-NEXT:  .long	3                       # 0x3
; ALL-NOT:   .long

define <16 x i32> @f16xi32_i128(<16 x i32> %a) {
; AVX2-LABEL: f16xi32_i128:
; AVX2:       # BB#0:
; AVX2-NEXT:    vbroadcasti128 {{.*#+}} ymm2 = mem[0,1,0,1]
; AVX2-NEXT:    vpaddd %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vpaddd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vpand %ymm2, %ymm1, %ymm1
;
; AVX512-LABEL: f16xi32_i128:
; AVX512:       # BB#0:
; AVX512-NEXT:    vbroadcasti32x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX512-NEXT:    vpaddd %zmm1, %zmm0, %zmm0
; AVX512-NEXT:    vpandd %zmm1, %zmm0, %zmm0
  %res1 = add <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>, %a
  %res2 = and <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>, %res1
  ret <16 x i32> %res2
}


; ALL64:       .LCPI26
; ALL64-NEXT:  .quad	0                       # 0x0
; ALL64-NEXT:  .quad	1                       # 0x1
; ALL64-NOT:   .quad

define <4 x i64> @f4xi64_i128(<4 x i64> %a) {
; ALL64-LABEL: f4xi64_i128:
; ALL64:       # BB#0:
; ALL64-NEXT:    vbroadcasti128 {{.*#+}} ymm1 = mem[0,1,0,1]
; ALL64-NEXT:    vpaddq %ymm1, %ymm0, %ymm0
; ALL64-NEXT:    vpand %ymm1, %ymm0, %ymm0
; ALL64-NEXT:    retq
  %res1 = add <4 x i64> <i64 0, i64 1, i64 0, i64 1>, %a
  %res2 = and <4 x i64> <i64 0, i64 1, i64 0, i64 1>, %res1
  ret <4 x i64> %res2
}


; ALL64:       .LCPI27
; ALL64-NEXT:  .quad	0                       # 0x0
; ALL64-NEXT:  .quad	1                       # 0x1
; ALL64-NOT:   .quad

define <8 x i64> @f8xi64_i128(<8 x i64> %a) {
; AVX2-64-LABEL: f8xi64_i128:
; AVX2-64:       # BB#0:
; AVX2-64-NEXT:    vbroadcasti128 {{.*#+}} ymm2 = mem[0,1,0,1]
; AVX2-64-NEXT:    vpaddq %ymm2, %ymm1, %ymm1
; AVX2-64-NEXT:    vpaddq %ymm2, %ymm0, %ymm0
; AVX2-64-NEXT:    vpand %ymm2, %ymm0, %ymm0
; AVX2-64-NEXT:    vpand %ymm2, %ymm1, %ymm1
; AVX2-64-NEXT:    retq
;
; AVX512F-64-LABEL: f8xi64_i128:
; AVX512F-64:       # BB#0:
; AVX512F-64-NEXT:    vbroadcasti32x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX512F-64-NEXT:    vpaddq %zmm1, %zmm0, %zmm0
; AVX512F-64-NEXT:    vpandq %zmm1, %zmm0, %zmm0
; AVX512F-64-NEXT:    retq
;
; AVX512BW-64-LABEL: f8xi64_i128:
; AVX512BW-64:       # BB#0:
; AVX512BW-64-NEXT:    vbroadcasti32x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX512BW-64-NEXT:    vpaddq %zmm1, %zmm0, %zmm0
; AVX512BW-64-NEXT:    vpandq %zmm1, %zmm0, %zmm0
; AVX512BW-64-NEXT:    retq
  %res1 = add <8 x i64> <i64 0, i64 1, i64 0, i64 1, i64 0, i64 1, i64 0, i64 1>, %a
  %res2 = and <8 x i64> <i64 0, i64 1, i64 0, i64 1, i64 0, i64 1, i64 0, i64 1>, %res1
  ret <8 x i64> %res2
}


; ALL64:            .LCPI28
; ALL64-NEXT:       .quad	0                       # 0x0
; ALL64-NEXT:       .quad	1                       # 0x1
; ALL64-NEXT:       .quad	2                       # 0x2
; ALL64-NEXT:       .quad	3                       # 0x3
; ALL64-NOT:        .quad

define <8 x i64> @f8xi64_i256(<8 x i64> %a) {
; AVX512F-64-LABEL: f8xi64_i256:
; AVX512F-64:       # BB#0:
; AVX512F-64-NEXT:    vbroadcasti64x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3]
; AVX512F-64-NEXT:    vpaddq %zmm1, %zmm0, %zmm0
; AVX512F-64-NEXT:    vpandq %zmm1, %zmm0, %zmm0
; AVX512F-64-NEXT:    retq
;
; AVX512BW-64-LABEL: f8xi64_i256:
; AVX512BW-64:       # BB#0:
; AVX512BW-64-NEXT:    vbroadcasti64x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3]
; AVX512BW-64-NEXT:    vpaddq %zmm1, %zmm0, %zmm0
; AVX512BW-64-NEXT:    vpandq %zmm1, %zmm0, %zmm0
; AVX512BW-64-NEXT:    retq
  %res1 = add <8 x i64> <i64 0, i64 1, i64 2, i64 3, i64 0, i64 1, i64 2, i64 3>, %a
  %res2 = and <8 x i64> <i64 0, i64 1, i64 2, i64 3, i64 0, i64 1, i64 2, i64 3>, %res1
  ret <8 x i64> %res2
}


; ALL:       .LCPI29
; ALL-NEXT:  .quad	4575657222482165760     

; AVX:       .LCPI29
; AVX-NEXT:  .quad	4575657222482165760     # double 0.0078125018626451492

define <4 x float> @f4xf32_f64(<4 x float> %a) {
; ALL32-LABEL: f4xf32_f64:
; ALL32:       # BB#0:
; ALL32-NEXT:    vmovddup {{.*#+}} xmm1 = mem[0,0]
; ALL32-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    vdivps %xmm0, %xmm1, %xmm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f4xf32_f64:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastq {{.*}}(%rip), %xmm1
; ALL64-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    vdivps %xmm0, %xmm1, %xmm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f4xf32_f64:
; AVX:       # BB#0:
; AVX-NEXT:    vmovddup {{.*#+}} xmm1 = mem[0,0]
; AVX-NEXT:    vaddps %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vdivps %xmm0, %xmm1, %xmm0
  %res1 = fadd <4 x float> <float 2.0, float 1.0, float 2.0, float 1.0>, %a
  %res2 = fdiv <4 x float> <float 2.0, float 1.0, float 2.0, float 1.0>, %res1
  ret <4 x float> %res2
}


; ALL64:       .LCPI30
; ALL64-NEXT:  .quad	4575657222482165760     # 0x3f80000040000000

; ALL32:         .LCPI30
; ALL32-NEXT:    .quad	4575657222482165760     # double 0.0078125018626451492

; AVX:         .LCPI30
; AVX-NEXT:    .quad	4575657222482165760     # double 0.0078125018626451492

define <8 x float> @f8xf32_f64(<8 x float> %a) {
; ALL-LABEL: f8xf32_f64:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastsd {{.*}}, %ymm1
; ALL-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vdivps %ymm0, %ymm1, %ymm0
;
; AVX-LABEL: f8xf32_f64:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastsd {{\.LCPI.*}}, %ymm1
; AVX-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vdivps %ymm0, %ymm1, %ymm0
  %res1 = fadd <8 x float> <float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0>, %a
  %res2 = fdiv <8 x float> <float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0>, %res1
  ret <8 x float> %res2
}


; ALL:       .LCPI31
; ALL-NEXT:  .long	1082130432              # float 4
; ALL-NEXT:  .long	1065353216              # float 1
; ALL-NEXT:  .long	1073741824              # float 2
; ALL-NEXT:  .long	1077936128              # float 3
; ALL-NOT:   .long

define <8 x float> @f8xf32_f128(<8 x float> %a) {
; ALL-LABEL: f8xf32_f128:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastf128 {{.*#+}} ymm1 = mem[0,1,0,1]
; ALL-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vdivps %ymm0, %ymm1, %ymm0
;
; AVX-LABEL: f8xf32_f128:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastf128 {{.*#+}} ymm1 = mem[0,1,0,1]
; AVX-NEXT:    vaddps %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vdivps %ymm0, %ymm1, %ymm0
  %res1 = fadd <8 x float> <float 4.0, float 1.0, float 2.0, float 3.0, float 4.0, float 1.0, float 2.0, float 3.0>, %a
  %res2 = fdiv <8 x float> <float 4.0, float 1.0, float 2.0, float 3.0, float 4.0, float 1.0, float 2.0, float 3.0>, %res1
  ret <8 x float> %res2
}


; ALL64:       .LCPI32
; ALL64-NEXT:  .quad	4575657222482165760     # 0x3f80000040000000

; ALL32:       .LCPI32
; ALL32-NEXT:  .quad	4575657222482165760     # double 0.0078125018626451492

; AVX:       .LCPI32
; AVX-NEXT:  .quad	4575657222482165760     # double 0.0078125018626451492

define <16 x float> @f16xf32_f64(<16 x float> %a) {
; AVX2-LABEL: f16xf32_f64:
; AVX2:       # BB#0:
; AVX2-NEXT:    vbroadcastsd {{.*}}, %ymm2
; AVX2-NEXT:    vaddps %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vaddps %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vdivps %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    vdivps %ymm1, %ymm2, %ymm1
;
; AVX512-LABEL: f16xf32_f64:
; AVX512:       # BB#0:
; AVX512-NEXT:    vbroadcastsd {{.*}}, %zmm1
; AVX512-NEXT:    vaddps %zmm1, %zmm0, %zmm0
; AVX512-NEXT:    vdivps %zmm0, %zmm1, %zmm0
;
; AVX-LABEL: f16xf32_f64:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastsd {{\.LCPI.*}}, %ymm2
; AVX-NEXT:    vaddps %ymm2, %ymm1, %ymm1
; AVX-NEXT:    vaddps %ymm2, %ymm0, %ymm0
; AVX-NEXT:    vdivps %ymm0, %ymm2, %ymm0
; AVX-NEXT:    vdivps %ymm1, %ymm2, %ymm1
  %res1 = fadd <16 x float> <float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0>, %a
  %res2 = fdiv <16 x float> <float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0, float 2.0, float 1.0>, %res1
  ret <16 x float> %res2
}


; ALL:       .LCPI33
; ALL-NEXT:  .long	1082130432              # float 4
; ALL-NEXT:  .long	1065353216              # float 1
; ALL-NEXT:  .long	1073741824              # float 2
; ALL-NEXT:  .long	1077936128              # float 3
; ALL-NOT:   .long

define <16 x float> @f16xf32_f128(<16 x float> %a) {
; AVX2-LABEL: f16xf32_f128:
; AVX2:       # BB#0:
; AVX2-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = mem[0,1,0,1]
; AVX2-NEXT:    vaddps %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vaddps %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vdivps %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    vdivps %ymm1, %ymm2, %ymm1
;
; AVX512-LABEL: f16xf32_f128:
; AVX512:       # BB#0:
; AVX512-NEXT:    vbroadcastf32x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX512-NEXT:    vaddps %zmm1, %zmm0, %zmm0
; AVX512-NEXT:    vdivps %zmm0, %zmm1, %zmm0
;
; AVX-LABEL: f16xf32_f128:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = mem[0,1,0,1]
; AVX-NEXT:    vaddps %ymm2, %ymm1, %ymm1
; AVX-NEXT:    vaddps %ymm2, %ymm0, %ymm0
; AVX-NEXT:    vdivps %ymm0, %ymm2, %ymm0
; AVX-NEXT:    vdivps %ymm1, %ymm2, %ymm1
  %res1 = fadd <16 x float> <float 4.0, float 1.0, float 2.0, float 3.0, float 4.0, float 1.0, float 2.0, float 3.0, float 4.0, float 1.0, float 2.0, float 3.0, float 4.0, float 1.0, float 2.0, float 3.0>, %a
  %res2 = fdiv <16 x float> <float 4.0, float 1.0, float 2.0, float 3.0, float 4.0, float 1.0, float 2.0, float 3.0, float 4.0, float 1.0, float 2.0, float 3.0, float 4.0, float 1.0, float 2.0, float 3.0>, %res1
  ret <16 x float> %res2
}


; AVX512:       .LCPI34
; AVX512-NEXT:  .long	1090519040              # float 8
; AVX512-NEXT:  .long	1065353216              # float 1
; AVX512-NEXT:  .long	1073741824              # float 2
; AVX512-NEXT:  .long	1077936128              # float 3
; AVX512-NEXT:  .long	1082130432              # float 4
; AVX512-NEXT:  .long	1084227584              # float 5
; AVX512-NEXT:  .long	1086324736              # float 6
; AVX512-NEXT:  .long	1088421888              # float 7
; AVX512-NOT:   .long

define <16 x float> @f16xf32_f256(<16 x float> %a) {
; AVX512-LABEL: f16xf32_f256:
; AVX512:       # BB#0:
; AVX512-NEXT:    vbroadcastf64x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3]
; AVX512-NEXT:    vaddps %zmm1, %zmm0, %zmm0
; AVX512-NEXT:    vdivps %zmm0, %zmm1, %zmm0
  %res1 = fadd <16 x float> <float 8.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0>, %a
  %res2 = fdiv <16 x float> <float 8.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0>, %res1
  ret <16 x float> %res2
}


; ALL:       .LCPI35
; ALL-NEXT:  .quad	4611686018427387904     # double 2
; ALL-NEXT:  .quad	4607182418800017408     # double 1
; ALL-NOT:   .quad

define <4 x double> @f4xf64_f128(<4 x double> %a) {
; ALL-LABEL: f4xf64_f128:
; ALL:       # BB#0:
; ALL-NEXT:    vbroadcastf128 {{.*#+}} ymm1 = mem[0,1,0,1]
; ALL-NEXT:    vaddpd %ymm1, %ymm0, %ymm0
; ALL-NEXT:    vdivpd %ymm0, %ymm1, %ymm0
;
; AVX-LABEL: f4xf64_f128:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastf128 {{.*#+}} ymm1 = mem[0,1,0,1]
; AVX-NEXT:    vaddpd %ymm1, %ymm0, %ymm0
; AVX-NEXT:    vdivpd %ymm0, %ymm1, %ymm0
  %res1 = fadd <4 x double> <double 2.0, double 1.0, double 2.0, double 1.0>, %a
  %res2 = fdiv <4 x double> <double 2.0, double 1.0, double 2.0, double 1.0>, %res1
  ret <4 x double> %res2
}


; ALL:       .LCPI36
; ALL-NEXT:  .quad	4611686018427387904     # double 2
; ALL-NEXT:  .quad	4607182418800017408     # double 1
; ALL-NOT:   .quad

define <8 x double> @f8xf64_f128(<8 x double> %a) {
; AVX2-LABEL: f8xf64_f128:
; AVX2:       # BB#0:
; AVX2-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = mem[0,1,0,1]
; AVX2-NEXT:    vaddpd %ymm2, %ymm1, %ymm1
; AVX2-NEXT:    vaddpd %ymm2, %ymm0, %ymm0
; AVX2-NEXT:    vdivpd %ymm0, %ymm2, %ymm0
; AVX2-NEXT:    vdivpd %ymm1, %ymm2, %ymm1
;
; AVX512-LABEL: f8xf64_f128:
; AVX512:       # BB#0:
; AVX512-NEXT:    vbroadcastf32x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]
; AVX512-NEXT:    vaddpd %zmm1, %zmm0, %zmm0
; AVX512-NEXT:    vdivpd %zmm0, %zmm1, %zmm0
;
; AVX-LABEL: f8xf64_f128:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastf128 {{.*#+}} ymm2 = mem[0,1,0,1]
; AVX-NEXT:    vaddpd %ymm2, %ymm1, %ymm1
; AVX-NEXT:    vaddpd %ymm2, %ymm0, %ymm0
; AVX-NEXT:    vdivpd %ymm0, %ymm2, %ymm0
; AVX-NEXT:    vdivpd %ymm1, %ymm2, %ymm1
  %res1 = fadd <8 x double> <double 2.0, double 1.0, double 2.0, double 1.0, double 2.0, double 1.0, double 2.0, double 1.0>, %a
  %res2 = fdiv <8 x double> <double 2.0, double 1.0, double 2.0, double 1.0, double 2.0, double 1.0, double 2.0, double 1.0>, %res1
  ret <8 x double> %res2
}


; AVX512:       .LCPI37
; AVX512-NEXT:  .quad	4616189618054758400     # double 4
; AVX512-NEXT:  .quad	4607182418800017408     # double 1
; AVX512-NEXT:  .quad	4611686018427387904     # double 2
; AVX512-NEXT:  .quad	4613937818241073152     # double 3
; AVX512-NOT:   .quad

define <8 x double> @f8xf64_f256(<8 x double> %a) {
; AVX512-LABEL: f8xf64_f256:
; AVX512:       # BB#0:
; AVX512-NEXT:    vbroadcastf64x4 {{.*#+}} zmm1 = mem[0,1,2,3,0,1,2,3]
; AVX512-NEXT:    vaddpd %zmm1, %zmm0, %zmm0
; AVX512-NEXT:    vdivpd %zmm0, %zmm1, %zmm0
  %res1 = fadd <8 x double> <double 4.0, double 1.0, double 2.0, double 3.0, double 4.0, double 1.0, double 2.0, double 3.0>, %a
  %res2 = fdiv <8 x double> <double 4.0, double 1.0, double 2.0, double 3.0, double 4.0, double 1.0, double 2.0, double 3.0>, %res1
  ret <8 x double> %res2
}



; ALL:       .LCPI38
; ALL-NEXT:  .long	4290379776              # 0xffba0000

; AVX:       .LCPI38
; AVX-NEXT:  .long	4290379776              # float NaN

define <8 x i16> @f8xi16_i32_NaN(<8 x i16> %a) {
; ALL32-LABEL: f8xi16_i32_NaN:
; ALL32:       # BB#0:
; ALL32-NEXT:    vpbroadcastd {{\.LCPI.*}}, %xmm1
; ALL32-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL32-NEXT:    retl
;
; ALL64-LABEL: f8xi16_i32_NaN:
; ALL64:       # BB#0:
; ALL64-NEXT:    vpbroadcastd {{.*}}(%rip), %xmm1
; ALL64-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    vpand %xmm1, %xmm0, %xmm0
; ALL64-NEXT:    retq
;
; AVX-LABEL: f8xi16_i32_NaN:
; AVX:       # BB#0:
; AVX-NEXT:    vbroadcastss {{\.LCPI.*}}, %xmm1
; AVX-NEXT:    vpaddw %xmm1, %xmm0, %xmm0
; AVX-NEXT:    vpand %xmm1, %xmm0, %xmm0
  %res1 = add <8 x i16> <i16 0, i16 -70, i16 0, i16 -70, i16 0, i16 -70, i16 0, i16 -70>, %a
  %res2 = and <8 x i16> <i16 0, i16 -70, i16 0, i16 -70, i16 0, i16 -70, i16 0, i16 -70>, %res1
  ret <8 x i16> %res2
}
