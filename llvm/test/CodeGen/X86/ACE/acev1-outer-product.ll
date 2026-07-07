; ACE v1 Outer Product Operations Test
; Tests all ACE outer product variants: BF16, INT8 (signed/unsigned combinations)
; Note: ACE v1 uses TILEMOVROW to extract tile data (not TILESTORED)
;
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+avx512f,+avx512bf16 -verify-machineinstrs | FileCheck %s

; Test BF16 outer product (TOP2BF16PS)
define void @test_top2bf16ps(ptr %out, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b) {
; CHECK-LABEL: test_top2bf16ps:
; CHECK:       ldtilecfg
; CHECK:       tilezero %tmm{{[0-7]}}
; CHECK:       top2bf16ps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)

  ; Extract using TILEMOVROW (ACE v1 pattern)
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test unsigned x unsigned -> dword (TOP4BUUD)
define void @test_top4buud(ptr %out, <64 x i8> %a_i8) {
; CHECK-LABEL: test_top4buud:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4buud
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4buud.internal(i16 16, i16 64, i16 64, x86_amx %c, <64 x i8> %a_i8, <64 x i8> %a_i8)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test signed x signed -> dword (TOP4BSSD)
define void @test_top4bssd(ptr %out, <64 x i8> %a_i8) {
; CHECK-LABEL: test_top4bssd:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4bssd
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %c, <64 x i8> %a_i8, <64 x i8> %a_i8)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test unsigned x signed -> dword (TOP4BUSD)
define void @test_top4busd(ptr %out, <64 x i8> %a_i8) {
; CHECK-LABEL: test_top4busd:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4busd
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4busd.internal(i16 16, i16 64, i16 64, x86_amx %c, <64 x i8> %a_i8, <64 x i8> %a_i8)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test signed x unsigned -> dword (TOP4BSUD)
define void @test_top4bsud(ptr %out, <64 x i8> %a_i8) {
; CHECK-LABEL: test_top4bsud:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4bsud
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4bsud.internal(i16 16, i16 64, i16 64, x86_amx %c, <64 x i8> %a_i8, <64 x i8> %a_i8)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test MX FP8 HF8xHF8 -> FP32 with BSR scaling (TOP4MXHF8PS)
define void @test_top4mxhf8ps_internal(ptr %out, <16 x i32> %zmm_a, <16 x i32> %zmm_b) {
; CHECK-LABEL: test_top4mxhf8ps_internal:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4mxhf8ps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4mxhf8ps.internal(i16 16, i16 64, i16 64, i8 5, x86_amx %c, <16 x i32> %zmm_a, <16 x i32> %zmm_b)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test MX FP8 BF8xHF8 -> FP32 with BSR scaling (TOP4MXBHF8PS)
define void @test_top4mxbhf8ps_internal(ptr %out, <16 x i32> %zmm_a, <16 x i32> %zmm_b) {
; CHECK-LABEL: test_top4mxbhf8ps_internal:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4mxbhf8ps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4mxbhf8ps.internal(i16 16, i16 64, i16 64, i8 3, x86_amx %c, <16 x i32> %zmm_a, <16 x i32> %zmm_b)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test MX FP8 HF8xBF8 -> FP32 with BSR scaling (TOP4MXHBF8PS)
define void @test_top4mxhbf8ps_internal(ptr %out, <16 x i32> %zmm_a, <16 x i32> %zmm_b) {
; CHECK-LABEL: test_top4mxhbf8ps_internal:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4mxhbf8ps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4mxhbf8ps.internal(i16 16, i16 64, i16 64, i8 7, x86_amx %c, <16 x i32> %zmm_a, <16 x i32> %zmm_b)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test MX FP8 BF8xBF8 -> FP32 with BSR scaling (TOP4MXBF8PS)
define void @test_top4mxbf8ps_internal(ptr %out, <16 x i32> %zmm_a, <16 x i32> %zmm_b) {
; CHECK-LABEL: test_top4mxbf8ps_internal:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4mxbf8ps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4mxbf8ps.internal(i16 16, i16 64, i16 64, i8 1, x86_amx %c, <16 x i32> %zmm_a, <16 x i32> %zmm_b)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test MX INT8 SxS -> FP32 with BSR scaling (TOP4MXBSSPS)
define void @test_top4mxbssps_internal(ptr %out, <16 x i32> %zmm_a, <16 x i32> %zmm_b) {
; CHECK-LABEL: test_top4mxbssps_internal:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top4mxbssps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.top4mxbssps.internal(i16 16, i16 64, i16 64, i8 9, x86_amx %c, <16 x i32> %zmm_a, <16 x i32> %zmm_b)

  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

; Test chained outer products (accumulation) - simplified to avoid spills
define void @test_chained_outer_products(ptr %out, <32 x bfloat> %zmm_bf16, <64 x i8> %a_i8) {
; CHECK-LABEL: test_chained_outer_products:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top2bf16ps
; CHECK:       top4bssd
; CHECK:       top4buud
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  ; BF16 outer product
  %d1 = call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c, <32 x bfloat> %zmm_bf16, <32 x bfloat> %zmm_bf16)

  ; Integer outer products
  %d2 = call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %d1, <64 x i8> %a_i8, <64 x i8> %a_i8)
  %d3 = call x86_amx @llvm.x86.top4buud.internal(i16 16, i16 64, i16 64, x86_amx %d2, <64 x i8> %a_i8, <64 x i8> %a_i8)

  ; Extract using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d3, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64
  ret void
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.top2bf16ps.internal(i16, i16, i16, x86_amx, <32 x bfloat>, <32 x bfloat>)
declare x86_amx @llvm.x86.top4buud.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.top4bssd.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.top4busd.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.top4bsud.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.top4mxhf8ps.internal(i16, i16, i16, i8, x86_amx, <16 x i32>, <16 x i32>)
declare x86_amx @llvm.x86.top4mxbhf8ps.internal(i16, i16, i16, i8, x86_amx, <16 x i32>, <16 x i32>)
declare x86_amx @llvm.x86.top4mxhbf8ps.internal(i16, i16, i16, i8, x86_amx, <16 x i32>, <16 x i32>)
declare x86_amx @llvm.x86.top4mxbf8ps.internal(i16, i16, i16, i8, x86_amx, <16 x i32>, <16 x i32>)
declare x86_amx @llvm.x86.top4mxbssps.internal(i16, i16, i16, i8, x86_amx, <16 x i32>, <16 x i32>)
declare <16 x i32> @llvm.x86.tilemovrow.internal(i16, i16, x86_amx, i32)
