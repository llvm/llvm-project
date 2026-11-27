; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+avx10.1 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+amx-tile -verify-machineinstrs | FileCheck %s

; Test basic ACE tile operations: tilezero and outer products. ACE v1 alone, and
; composed with AMX-TILE, must both work. ACE v1 extracts tile data with
; TILEMOVROW rather than TILESTORED.

define void @test_acev1_basic(ptr %out, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b) {
; CHECK-LABEL: test_acev1_basic:
; CHECK:       ldtilecfg
; CHECK:       tilezero %tmm{{[0-7]}}
; CHECK:       top2bf16ps
; CHECK:       tilemovrow
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)

  ; ACE BF16 outer product
  %d = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)

  ; Extract result using TILEMOVROW (ACE v1 pattern - no TILESTORED)
  %row0 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

; The ACE spelling of the shared row-read intrinsic selects its own pseudo, so
; the palette follows from the opcode rather than from the subtarget.
define void @test_acev1_cvtrowd2ps_internal(ptr %out, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b, i32 %idx) {
; CHECK-LABEL: test_acev1_cvtrowd2ps_internal:
; CHECK:       ldtilecfg
; CHECK:       tilezero %tmm{{[0-7]}}
; CHECK:       top2bf16ps
; CHECK:       tcvtrowd2ps
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)
  %ps = call <16 x float> @llvm.x86.acev1.tcvtrowd2ps.internal(i16 16, i16 64, x86_amx %d, i32 %idx)
  store volatile <16 x float> %ps, ptr %out, align 64
  ret void
}

; A tile zero on its own must still establish the palette: it is the only tile
; instruction here, so nothing else can set the program model for it.
define void @test_acev1_tilezero_only(ptr %out) {
; CHECK-LABEL: test_acev1_tilezero_only:
; CHECK:       movb $2,
; CHECK:       ldtilecfg
; CHECK:       tilezero %tmm{{[0-7]}}
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %v = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %c)
  store volatile <256 x i32> %v, ptr %out, align 64
  ret void
}

; The remaining shared row reads take the same ACE-spelled struct-API path.
define void @test_acev1_cvtrowps2_internal(ptr %out, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b, i32 %idx) {
; CHECK-LABEL: test_acev1_cvtrowps2_internal:
; CHECK:       ldtilecfg
; CHECK:       tcvtrowps2bf16h
; CHECK:       tcvtrowps2bf16l
; CHECK:       tcvtrowps2phh
; CHECK:       tcvtrowps2phl
; CHECK:       tilerelease
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)
  %r0 = call <32 x bfloat> @llvm.x86.acev1.tcvtrowps2bf16h.internal(i16 16, i16 64, x86_amx %d, i32 %idx)
  store volatile <32 x bfloat> %r0, ptr %out, align 64
  %r1 = call <32 x bfloat> @llvm.x86.acev1.tcvtrowps2bf16l.internal(i16 16, i16 64, x86_amx %d, i32 %idx)
  store volatile <32 x bfloat> %r1, ptr %out, align 64
  %r2 = call <32 x half> @llvm.x86.acev1.tcvtrowps2phh.internal(i16 16, i16 64, x86_amx %d, i32 %idx)
  store volatile <32 x half> %r2, ptr %out, align 64
  %r3 = call <32 x half> @llvm.x86.acev1.tcvtrowps2phl.internal(i16 16, i16 64, x86_amx %d, i32 %idx)
  store volatile <32 x half> %r3, ptr %out, align 64
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
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)

  ; Unsigned x Unsigned -> DWORD
  %d0 = call x86_amx @llvm.x86.acev1.top4buud.internal(i16 16, i16 64, x86_amx %c, <64 x i8> %a_i8, <64 x i8> %b_i8)

  ; Signed x Signed -> DWORD
  %d1 = call x86_amx @llvm.x86.acev1.top4bssd.internal(i16 16, i16 64, x86_amx %d0, <64 x i8> %a_i8, <64 x i8> %b_i8)

  ; Extract result using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d1, i32 0)
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
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)

  ; Multiple accumulations
  %d0 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %c, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)
  %d1 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %d0, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)
  %d2 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %d1, <32 x bfloat> %zmm_a, <32 x bfloat> %zmm_b)

  ; Extract result using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d2, i32 0)
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
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)

  ; Extract using TILEMOVROW
  %row0 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %c, i32 0)
  store volatile <16 x i32> %row0, ptr %out, align 64

  ret void
}

declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16, i16, x86_amx, <32 x bfloat>, <32 x bfloat>)
declare x86_amx @llvm.x86.acev1.top4buud.internal(i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.acev1.top4bssd.internal(i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16, i16, x86_amx, i32)
declare <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx)
declare <16 x float> @llvm.x86.acev1.tcvtrowd2ps.internal(i16, i16, x86_amx, i32)
declare <32 x bfloat> @llvm.x86.acev1.tcvtrowps2bf16h.internal(i16, i16, x86_amx, i32)
declare <32 x bfloat> @llvm.x86.acev1.tcvtrowps2bf16l.internal(i16, i16, x86_amx, i32)
declare <32 x half> @llvm.x86.acev1.tcvtrowps2phh.internal(i16, i16, x86_amx, i32)
declare <32 x half> @llvm.x86.acev1.tcvtrowps2phl.internal(i16, i16, x86_amx, i32)
