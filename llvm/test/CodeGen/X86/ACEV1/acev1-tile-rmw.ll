; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 \
; RUN:   -verify-machineinstrs -stop-after=x86-fast-pre-tile-config -o - \
; RUN:   | FileCheck %s --check-prefix=MIR
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=ASM,O2

; Test the tile row and column writes modelled as read-modify-write: they write
; one row or column and keep the rest, so the destination is a source as well.

; A tile used in another block is spilled and reloaded, so rows move both ways.
define void @reload_is_a_merge_chain(ptr %out, i1 %c, <32 x bfloat> %a, <32 x bfloat> %b) nounwind {
; The spill reads the rows out first, into independent ZMM defs.
; MIR-LABEL: name: reload_is_a_merge_chain
; MIR:         %{{[0-9]+}}:vr512 = TILEMOVROWrti %{{[0-9]+}}, 0
;
; The reload threads one register through the rows; row 0 starts from a def.
; MIR:         [[SEED:%[0-9]+]]:tile = IMPLICIT_DEF
; MIR:         [[R0:%[0-9]+]]:tile = TILEMOVROWtri [[SEED]], %{{[0-9]+}}, 0
; MIR-NEXT:    {{.*}}VMOVUPSZrm
; MIR-NEXT:    [[R1:%[0-9]+]]:tile = TILEMOVROWtri [[R0]], %{{[0-9]+}}, 1
; MIR-NEXT:    {{.*}}VMOVUPSZrm
; MIR-NEXT:    %{{[0-9]+}}:tile = TILEMOVROWtri [[R1]], %{{[0-9]+}}, 2
entry:
  %t = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %d = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t, <32 x bfloat> %a, <32 x bfloat> %b)
  br i1 %c, label %use, label %exit

use:
  %r = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d, i32 0)
  store <16 x i32> %r, ptr %out
  br label %exit

exit:
  ret void
}

; A row write merges into the tile it is given, so the tile it writes and the
; tile it returns are the same one.
define void @write_merges_into_the_given_tile(ptr %out, <16 x i32> %v, i32 %idx) nounwind {
; ASM-LABEL: write_merges_into_the_given_tile:
; ASM:         ldtilecfg
; ASM:         tilezero %[[T:tmm[0-7]]]
; ASM:         tilemovrow %{{[a-z0-9]+}}, %zmm{{[0-9]+}}, %[[T]]
; ASM:         tilemovcol %{{[a-z0-9]+}}, %zmm{{[0-9]+}}, %[[T]]
entry:
  %t = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %row = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %t, <16 x i32> %v, i32 %idx)
  %col = call x86_amx @llvm.x86.acev1.tilemovcolinsert.internal(i16 16, i16 64, x86_amx %row, <16 x i32> %v, i32 %idx)
  %a = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %col, i32 0)
  store <16 x i32> %a, ptr %out
  ret void
}

; Two tiles written alternately have to stay in registers of their own, or the
; second write to %t0 lands in the tile %t1 is using.
define void @interleaved_writes_keep_their_own_tiles(ptr %out0, ptr %out1, <16 x i32> %a, <16 x i32> %b, <16 x i32> %c) nounwind {
; At -O0 both tiles live on the stack, so only the optimised run says anything
; about which register each one gets. The registers are spelled out rather than
; captured because the point of the test is that the two tiles differ.
; O2-LABEL: interleaved_writes_keep_their_own_tiles:
; O2:         ldtilecfg
; O2:         tilezero %tmm0
; O2-NEXT:    tilezero %tmm1
; O2:         tilemovrow $0, %zmm0, %tmm0
; O2-NEXT:    tilemovrow $0, %zmm1, %tmm1
; O2:         tilemovrow $1, %zmm2, %tmm0
; O2-NEXT:    tilemovrow $0, %tmm0, %zmm0
; O2-NEXT:    tilemovrow $0, %tmm1, %zmm1
entry:
  %z0 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %z1 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t0 = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %z0, <16 x i32> %a, i32 0)
  %t1 = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %z1, <16 x i32> %b, i32 0)
  %t2 = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %t0, <16 x i32> %c, i32 1)
  ; Row 0 of %t2 still holds %a, and row 0 of %t1 still holds %b.
  %r0 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %t2, i32 0)
  %r1 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %t1, i32 0)
  store <16 x i32> %r0, ptr %out0
  store <16 x i32> %r1, ptr %out1
  ret void
}

; One intrinsic covers both index forms, so a constant must reach the imm one.
define void @constant_index_uses_the_immediate_form(ptr %out, <16 x i32> %v, i32 %idx) nounwind {
; ASM-LABEL: constant_index_uses_the_immediate_form:
; ASM:         tilemovrow $9, %zmm{{[0-9]+}}, %[[T:tmm[0-7]]]
; ASM:         tilemovcol $3, %zmm{{[0-9]+}}, %[[T]]
; ASM:         tilemovrow %{{[a-z0-9]+}}, %zmm{{[0-9]+}}, %[[T]]
entry:
  %t = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %row = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %t, <16 x i32> %v, i32 9)
  %col = call x86_amx @llvm.x86.acev1.tilemovcolinsert.internal(i16 16, i16 64, x86_amx %row, <16 x i32> %v, i32 3)
  %var = call x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16 16, i16 64, x86_amx %col, <16 x i32> %v, i32 %idx)
  %a = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %var, i32 0)
  store <16 x i32> %a, ptr %out
  ret void
}

declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16, i16, x86_amx, <32 x bfloat>, <32 x bfloat>)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16, i16, x86_amx, i32)
declare x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16, i16, x86_amx, <16 x i32>, i32)
declare x86_amx @llvm.x86.acev1.tilemovcolinsert.internal(i16, i16, x86_amx, <16 x i32>, i32)
