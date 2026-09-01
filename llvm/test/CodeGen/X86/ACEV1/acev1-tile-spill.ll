; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefix=ACE \
; RUN:     --implicit-check-not=tilestored --implicit-check-not=tileloadd
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1,+amx-tile -verify-machineinstrs \
; RUN:   | FileCheck %s --check-prefix=ACE \
; RUN:     --implicit-check-not=tilestored --implicit-check-not=tileloadd

; Test tile spilling. Nine simultaneously live tiles exceed the eight physical
; TMM registers, so the tile register allocator must spill. ACE v1 configures
; palette 2, which has no TILELOADD/TILESTORED, so a spill moves one row at a
; time through a scratch ZMM. ACE v1 alone, and composed with AMX-TILE, must
; both avoid TILELOADD/TILESTORED: the palette is 2 whenever ACE is enabled, and
; those forms are unavailable there.
;
; tilemovrow prints the same mnemonic in both directions; the operand order
; distinguishes them ($idx, %tmm, %zmm reads a row out, $idx, %zmm, %tmm writes
; one in).

; Nine live tiles accumulated with bf16 outer products.
define void @spill_nine_tiles_bf16(ptr %out, <32 x bfloat> %a, <32 x bfloat> %b) nounwind {
; ACE-LABEL: spill_nine_tiles_bf16:
; ACE:         ldtilecfg
; Spill reads each of the 16 rows into a ZMM and stores it to the stack slot.
; ACE-COUNT-16:  tilemovrow ${{[0-9]+}}, %tmm{{[0-7]}}, %zmm{{[0-9]+}}
; Reload loads each row back and writes it into the tile.
; ACE-COUNT-16:  tilemovrow ${{[0-9]+}}, %zmm{{[0-9]+}}, %tmm{{[0-7]}}
entry:
  %t0 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t1 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t2 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t3 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t4 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t5 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t6 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t7 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t8 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  %d0 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t0, <32 x bfloat> %a, <32 x bfloat> %b)
  %d1 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t1, <32 x bfloat> %a, <32 x bfloat> %b)
  %d2 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t2, <32 x bfloat> %a, <32 x bfloat> %b)
  %d3 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t3, <32 x bfloat> %a, <32 x bfloat> %b)
  %d4 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t4, <32 x bfloat> %a, <32 x bfloat> %b)
  %d5 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t5, <32 x bfloat> %a, <32 x bfloat> %b)
  %d6 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t6, <32 x bfloat> %a, <32 x bfloat> %b)
  %d7 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t7, <32 x bfloat> %a, <32 x bfloat> %b)
  %d8 = tail call x86_amx @llvm.x86.top2bf16ps.internal(i16 16, i16 64, i16 64, x86_amx %t8, <32 x bfloat> %a, <32 x bfloat> %b)

  ; Keep all nine results live to the end so the allocator has to spill.
  %r0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d0, i32 0)
  store volatile <16 x i32> %r0, ptr %out, align 64
  %r1 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d1, i32 0)
  store volatile <16 x i32> %r1, ptr %out, align 64
  %r2 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d2, i32 0)
  store volatile <16 x i32> %r2, ptr %out, align 64
  %r3 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d3, i32 0)
  store volatile <16 x i32> %r3, ptr %out, align 64
  %r4 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d4, i32 0)
  store volatile <16 x i32> %r4, ptr %out, align 64
  %r5 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d5, i32 0)
  store volatile <16 x i32> %r5, ptr %out, align 64
  %r6 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d6, i32 0)
  store volatile <16 x i32> %r6, ptr %out, align 64
  %r7 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d7, i32 0)
  store volatile <16 x i32> %r7, ptr %out, align 64
  %r8 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d8, i32 0)
  store volatile <16 x i32> %r8, ptr %out, align 64

  ret void
}

; Same pressure, integer outer products, to check the spill path is independent
; of the producing instruction.
define void @spill_nine_tiles_int(ptr %out, <64 x i8> %a) nounwind {
; ACE-LABEL: spill_nine_tiles_int:
; ACE:         ldtilecfg
; ACE-COUNT-16:  tilemovrow ${{[0-9]+}}, %tmm{{[0-7]}}, %zmm{{[0-9]+}}
; ACE-COUNT-16:  tilemovrow ${{[0-9]+}}, %zmm{{[0-9]+}}, %tmm{{[0-7]}}
entry:
  %t0 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t1 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t2 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t3 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t4 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t5 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t6 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t7 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %t8 = tail call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)

  %d0 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t0, <64 x i8> %a, <64 x i8> %a)
  %d1 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t1, <64 x i8> %a, <64 x i8> %a)
  %d2 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t2, <64 x i8> %a, <64 x i8> %a)
  %d3 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t3, <64 x i8> %a, <64 x i8> %a)
  %d4 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t4, <64 x i8> %a, <64 x i8> %a)
  %d5 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t5, <64 x i8> %a, <64 x i8> %a)
  %d6 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t6, <64 x i8> %a, <64 x i8> %a)
  %d7 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t7, <64 x i8> %a, <64 x i8> %a)
  %d8 = tail call x86_amx @llvm.x86.top4bssd.internal(i16 16, i16 64, i16 64, x86_amx %t8, <64 x i8> %a, <64 x i8> %a)

  %r0 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d0, i32 0)
  store volatile <16 x i32> %r0, ptr %out, align 64
  %r1 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d1, i32 0)
  store volatile <16 x i32> %r1, ptr %out, align 64
  %r2 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d2, i32 0)
  store volatile <16 x i32> %r2, ptr %out, align 64
  %r3 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d3, i32 0)
  store volatile <16 x i32> %r3, ptr %out, align 64
  %r4 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d4, i32 0)
  store volatile <16 x i32> %r4, ptr %out, align 64
  %r5 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d5, i32 0)
  store volatile <16 x i32> %r5, ptr %out, align 64
  %r6 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d6, i32 0)
  store volatile <16 x i32> %r6, ptr %out, align 64
  %r7 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d7, i32 0)
  store volatile <16 x i32> %r7, ptr %out, align 64
  %r8 = call <16 x i32> @llvm.x86.tilemovrow.internal(i16 16, i16 64, x86_amx %d8, i32 0)
  store volatile <16 x i32> %r8, ptr %out, align 64

  ret void
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.top2bf16ps.internal(i16, i16, i16, x86_amx, <32 x bfloat>, <32 x bfloat>)
declare x86_amx @llvm.x86.top4bssd.internal(i16, i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare <16 x i32> @llvm.x86.tilemovrow.internal(i16, i16, x86_amx, i32)
