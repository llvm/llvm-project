; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 \
; RUN:   -verify-machineinstrs -stop-after=x86-fast-pre-tile-config -o - \
; RUN:   | FileCheck %s --check-prefix=MIR
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=ASM
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=ASM,O2

; Test the tile row and column writes modelled as read-modify-write: they write
; one row or column and keep the rest, so the destination is a source as well.

; Nine live tiles exceed the eight physical TMM registers, forcing a spill.
define void @reload_is_a_merge_chain(ptr %out, <32 x bfloat> %a, <32 x bfloat> %b) nounwind {
; The reload threads one register through the rows; row 0 starts from a def.
; MIR-LABEL: name: reload_is_a_merge_chain
; MIR:         [[SEED:%[0-9]+]]:tile = IMPLICIT_DEF
; MIR:         [[R0:%[0-9]+]]:tile = TILEMOVROWtri [[SEED]], %{{[0-9]+}}, 0
; MIR-NEXT:    {{.*}}VMOVUPSZrm
; MIR-NEXT:    [[R1:%[0-9]+]]:tile = TILEMOVROWtri [[R0]], %{{[0-9]+}}, 1
; MIR-NEXT:    {{.*}}VMOVUPSZrm
; MIR-NEXT:    %{{[0-9]+}}:tile = TILEMOVROWtri [[R1]], %{{[0-9]+}}, 2
;
; The read direction writes a whole ZMM, so those defs stay independent.
; MIR:         %{{[0-9]+}}:vr512 = TILEMOVROWrti %{{[0-9]+}}, 0
entry:
  %t0 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t1 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t2 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t3 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t4 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t5 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t6 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t7 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t8 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)

  %d0 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t0, <32 x bfloat> %a, <32 x bfloat> %b)
  %d1 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t1, <32 x bfloat> %a, <32 x bfloat> %b)
  %d2 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t2, <32 x bfloat> %a, <32 x bfloat> %b)
  %d3 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t3, <32 x bfloat> %a, <32 x bfloat> %b)
  %d4 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t4, <32 x bfloat> %a, <32 x bfloat> %b)
  %d5 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t5, <32 x bfloat> %a, <32 x bfloat> %b)
  %d6 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t6, <32 x bfloat> %a, <32 x bfloat> %b)
  %d7 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t7, <32 x bfloat> %a, <32 x bfloat> %b)
  %d8 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %t8, <32 x bfloat> %a, <32 x bfloat> %b)

  ; Read a row back out of every tile to keep them all live across the region.
  %r0 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d0, i32 0)
  %r1 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d1, i32 0)
  %r2 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d2, i32 0)
  %r3 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d3, i32 0)
  %r4 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d4, i32 0)
  %r5 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d5, i32 0)
  %r6 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d6, i32 0)
  %r7 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d7, i32 0)
  %r8 = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %d8, i32 0)

  %s0 = add <16 x i32> %r0, %r1
  %s1 = add <16 x i32> %s0, %r2
  %s2 = add <16 x i32> %s1, %r3
  %s3 = add <16 x i32> %s2, %r4
  %s4 = add <16 x i32> %s3, %r5
  %s5 = add <16 x i32> %s4, %r6
  %s6 = add <16 x i32> %s5, %r7
  %s7 = add <16 x i32> %s6, %r8
  store <16 x i32> %s7, ptr %out
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
; O2:         tilemovrow %{{[a-z0-9]+}}, %zmm0, %tmm0
; O2-NEXT:    tilemovrow %{{[a-z0-9]+}}, %zmm1, %tmm1
; O2:         tilemovrow %{{[a-z0-9]+}}, %zmm2, %tmm0
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

declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16, i16, x86_amx, <32 x bfloat>, <32 x bfloat>)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16, i16, x86_amx, i32)
declare x86_amx @llvm.x86.acev1.tilemovrowinsert.internal(i16, i16, x86_amx, <16 x i32>, i32)
declare x86_amx @llvm.x86.acev1.tilemovcolinsert.internal(i16, i16, x86_amx, <16 x i32>, i32)
