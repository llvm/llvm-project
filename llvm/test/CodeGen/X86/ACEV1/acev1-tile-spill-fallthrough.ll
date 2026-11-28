; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+amx-tile -O0 \
; RUN:   -verify-machineinstrs | FileCheck %s

; Test spilling a tile that is defined by the last instruction of a block, where
; the insertion point one past the def is MBB.end().

; The tile in each arm is live out through the PHI, and the branch to the
; fall-through successor is gone by then, so the tile def ends the block.
define void @spill_tile_defined_last_in_block(ptr %out, i32 %cond, <32 x bfloat> %a, <32 x bfloat> %b) nounwind {
; CHECK-LABEL: spill_tile_defined_last_in_block:
; CHECK:         ldtilecfg
; CHECK-COUNT-16:  tilemovrow ${{[0-9]+}}, %tmm{{[0-7]}}, %zmm{{[0-9]+}}
; CHECK:         tilerelease
entry:
  %z = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %c = icmp eq i32 %cond, 0
  br i1 %c, label %then, label %else

then:
  %d1 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %z, <32 x bfloat> %a, <32 x bfloat> %b)
  br label %merge

else:
  %d2 = call x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16 16, i16 64, x86_amx %z, <32 x bfloat> %b, <32 x bfloat> %a)
  br label %merge

merge:
  %t = phi x86_amx [ %d1, %then ], [ %d2, %else ]
  %row = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %t, i32 0)
  store volatile <16 x i32> %row, ptr %out, align 64
  ret void
}

declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.acev1.top2bf16ps.internal(i16, i16, x86_amx, <32 x bfloat>, <32 x bfloat>)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16, i16, x86_amx, i32)
