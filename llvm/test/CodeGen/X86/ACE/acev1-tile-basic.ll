; ACE v1 Basic Tile Operations Test
; Tests basic ACE tile operations: tilezero, outer products
; Note: ACE v1 uses TILEMOVROW to extract tile data (not TILESTORED)
;
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+avx512f,+avx512bf16 -verify-machineinstrs | FileCheck %s

define void @test_acev1_basic(ptr %out, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b) {
; CHECK-LABEL: test_acev1_basic:
; CHECK:       ldtilecfg
; CHECK:       tilezero %tmm{{[0-7]}}
; CHECK:       top2bf16ps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  ; ACE BF16 outer product
  %d = call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)

  ; Extract result using TILEMOVROW (ACE v1 pattern - no TILESTORED)
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

; Test ACE integer outer products
define void @test_acev1_int_outer_products(ptr %out, <64 x i8> %a_i8, <64 x i8> %b_i8) {
; CHECK-LABEL: test_acev1_int_outer_products:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK-DAG:   top4buud
; CHECK-DAG:   top4bssd
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  ; Unsigned x Unsigned -> DWORD
  %d0 = call x86_amx @llvm.x86.top4buud.internal(i16 16, i16 64, i16 64, x86_amx %c, <64 x i8> %a_i8, <64 x i8> %b_i8)

  ; Signed x Signed -> DWORD
  %d1 = call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %d0, <64 x i8> %a_i8, <64 x i8> %b_i8)

  ; Extract result using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d1, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

; Test multiple outer product accumulations
define void @test_acev1_accumulation(ptr %out, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b) {
; CHECK-LABEL: test_acev1_accumulation:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       top2bf16ps
; CHECK:       top2bf16ps
; CHECK:       top2bf16ps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  ; Multiple accumulations
  %d0 = call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)
  %d1 = call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %d0, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)
  %d2 = call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %d1, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)

  ; Extract result using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d2, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

; Test basic tilezero operation
define void @test_acev1_tilezero(ptr %out) {
; CHECK-LABEL: test_acev1_tilezero:
; CHECK:       ldtilecfg
; CHECK:       tilezero
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  ; Extract using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %c, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.top2bf16ps.internal(i16, i16, i16, x86_amx, <32 x bfloat>, <32 x bfloat>)
declare x86_amx @llvm.x86.top4buud.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.top4bssd.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare <16 x i32> @llvm.x86.tilemovrow.internal(i16, i16, x86_amx, i32)
