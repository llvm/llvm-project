; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 \
; RUN:   -verify-machineinstrs | FileCheck %s \
; RUN:   --implicit-check-not=tilestored --implicit-check-not=tileloadd
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 \
; RUN:   -verify-machineinstrs | FileCheck %s \
; RUN:   --implicit-check-not=tilestored --implicit-check-not=tileloadd

; Test that tile data is not made volatile at -O0. X86VolatileTileData shortens
; tile live ranges before fast register allocation by round-tripping every tile
; through memory with TILESTORED/TILELOADD. ACE configures palette 2, where
; those are unavailable, so the transform must be skipped and no tile load/store
; may appear. -O2 is included as a control: the same IR must be clean at both
; levels.
;
; Only one tile is live per function here: with the tile data left non-volatile,
; anything that keeps a second tile live at -O0 reaches the row-by-row spill in
; X86FastPreTileConfig, which does not yet produce valid SSA. Extend this test
; once that is fixed.

; A tile is produced and read back a row at a time, never stored as a tile.
define void @extract_one_row(ptr %out) nounwind {
; CHECK-LABEL: extract_one_row:
; CHECK:         movb $2, {{-?[0-9]+}}(%rsp)
; CHECK:         ldtilecfg
; CHECK:         tilezero %tmm{{[0-7]}}
; CHECK:         tilemovrow ${{[0-9]+}}, %tmm{{[0-7]}}, %zmm{{[0-9]+}}
entry:
  %t = tail call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %r = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %t, i32 0)
  store volatile <16 x i32> %r, ptr %out, align 64
  ret void
}

; Several rows of the same tile, so the tile stays live across extractions.
define void @extract_several_rows(ptr %out) nounwind {
; CHECK-LABEL: extract_several_rows:
; CHECK:         movb $2, {{-?[0-9]+}}(%rsp)
; CHECK:         ldtilecfg
; CHECK:         tilezero %tmm{{[0-7]}}
; CHECK:         tilemovrow ${{[0-9]+}}, %tmm{{[0-7]}}, %zmm{{[0-9]+}}
; CHECK:         tilemovrow ${{[0-9]+}}, %tmm{{[0-7]}}, %zmm{{[0-9]+}}
entry:
  %t = tail call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %r0 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %t, i32 0)
  store volatile <16 x i32> %r0, ptr %out, align 64
  %r7 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %t, i32 7)
  store volatile <16 x i32> %r7, ptr %out, align 64
  ret void
}

; The converting row reads are gated the same way and must also stay clean.
define void @extract_row_convert(ptr %out) nounwind {
; CHECK-LABEL: extract_row_convert:
; CHECK:         movb $2, {{-?[0-9]+}}(%rsp)
; CHECK:         ldtilecfg
; CHECK:         tilezero %tmm{{[0-7]}}
; CHECK:         tcvtrowd2ps ${{[0-9]+}}, %tmm{{[0-7]}}, %zmm{{[0-9]+}}
entry:
  %t = tail call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %r = call <16 x float> @llvm.x86.acev1.tcvtrowd2ps.internal(i16 16, i16 64, x86_amx %t, i32 0)
  store volatile <16 x float> %r, ptr %out, align 64
  ret void
}

; ldtilecfg resets the BSR, so the config has to precede a BSR write.
define void @config_precedes_a_bsr_write(ptr %out, <64 x i8> %hi, <64 x i8> %lo) nounwind {
; CHECK-LABEL: config_precedes_a_bsr_write:
; CHECK:         ldtilecfg
; CHECK:         bsrmovf
; CHECK:         tilezero %tmm{{[0-7]}}
; CHECK:         top4mxbf8ps
entry:
  call void @llvm.x86.acev1.bsr0movf(<64 x i8> %hi, <64 x i8> %lo)
  %t = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.acev1.top4mxbf8ps.internal(i16 16, i16 64, i8 0, x86_amx %t, <64 x i8> %hi, <64 x i8> %lo)
  %r = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store volatile <16 x i32> %r, ptr %out, align 64
  ret void
}

declare void @llvm.x86.acev1.bsr0movf(<64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.acev1.top4mxbf8ps.internal(i16, i16, i8, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16, i16, x86_amx, i32)
declare <16 x float> @llvm.x86.acev1.tcvtrowd2ps.internal(i16, i16, x86_amx, i32)
