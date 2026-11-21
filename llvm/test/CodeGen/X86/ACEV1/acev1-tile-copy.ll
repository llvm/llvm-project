; NOTE: With no TILELOADD/TILESTORED, a tile-to-tile copy goes via a slot.
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+amx-tile -verify-machineinstrs | FileCheck %s

declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16, i16, x86_amx, <16 x i32>, i32)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16, i16, x86_amx, i32)

; A tile written twice needs a copy, so the rows move in both directions.
define void @copy_then_modify(<16 x i32> %v, ptr %o1, ptr %o2) nounwind {
; CHECK-LABEL: copy_then_modify:
; The copy reads all 16 rows out to the slot ...
; CHECK-COUNT-16:    tilemovrow ${{[0-9]+}}, %tmm{{[0-7]}}, %zmm{{[0-9]+}}
; CHECK:             vmovups {{[0-9]+}}(%rsp), %zmm{{[0-9]+}}
; ... and writes all 16 back into the other tile.
; CHECK-COUNT-16:    tilemovrow ${{[0-9]+}}, %zmm{{[0-9]+}}, %tmm{{[0-7]}}
  %z = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %a = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %z, <16 x i32> %v, i32 2)
  %b = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %a, <16 x i32> %v, i32 3)
  %ra = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %a, i32 2)
  %rb = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %b, i32 3)
  store <16 x i32> %ra, ptr %o1
  store <16 x i32> %rb, ptr %o2
  ret void
}
