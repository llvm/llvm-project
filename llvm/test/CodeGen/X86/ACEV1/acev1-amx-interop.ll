; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-tile,+amx-int8 \
; RUN:   -verify-machineinstrs | FileCheck %s --implicit-check-not=tilemovrow
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-tile,+amx-int8,+acev1 \
; RUN:   -verify-machineinstrs | FileCheck %s --implicit-check-not=tilemovrow
; The volatile tile model only runs at -O0, so that is the only level at which
; an ACE-aware subtarget could divert plain AMX tiles away from it.
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-tile,+amx-int8 -O0 \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=O0 \
; RUN:   --implicit-check-not=tilemovrow
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-tile,+amx-int8,+acev1 \
; RUN:   -O0 -verify-machineinstrs | FileCheck %s --check-prefix=O0 \
; RUN:   --implicit-check-not=tilemovrow

; Test that a function holding only plain AMX intrinsics lowers the same way
; whether or not the target also supports ACE v1.

define void @amx_load_store_on_ace_target(ptr %in, ptr %out) nounwind {
; CHECK-LABEL: amx_load_store_on_ace_target:
; CHECK:         movb $1, {{.*}}(%rsp)
; CHECK:         ldtilecfg
; CHECK:         tileloadd (%rdi,%rax), %tmm0
; CHECK-NEXT:    tilestored %tmm0, (%rsi,%rax)
; CHECK:         tilerelease
  %v = load <256 x i32>, ptr %in, align 64
  %t = call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> %v)
  call void @llvm.x86.tilestored64.internal(i16 16, i16 64, ptr %out, i64 64, x86_amx %t)
  ret void
}

; At -O0 the tile has to go through the volatile model, which stores it with
; TILESTORED and reads it straight back with TILELOADD. An ACE-aware subtarget
; used to skip that for AMX tiles too, leaving the tile register live instead.
define <256 x i32> @amx_tile_to_vector_on_ace_target() nounwind {
; CHECK-LABEL: amx_tile_to_vector_on_ace_target:
; CHECK:         movb $1, {{.*}}(%rsp)
; CHECK:         tilezero %tmm0
; CHECK:         tilestored %tmm0, (%rdi,%rsi)
; O0-LABEL: amx_tile_to_vector_on_ace_target:
; O0:         tilezero %tmm0
; O0:         tilestored %tmm0, (%rsi,%r8)
; O0:         tileloadd (%rsi,%r8), %tmm0
  %t = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %v = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %t)
  ret <256 x i32> %v
}

; Nine live tiles force a spill. These palette 1 tiles are 8 rows by 32 bytes,
; so they spill with TILESTORED rather than the row-by-row TILEMOVROW sequence.
define void @amx_spill_on_ace_target(ptr %out) nounwind {
; CHECK-LABEL: amx_spill_on_ace_target:
; CHECK:         tilestored %tmm{{[0-7]}}, {{.*}}(%r{{[a-z0-9]+}}
; CHECK:         tileloadd {{.*}}(%r{{[a-z0-9]+}}{{.*}}, %tmm{{[0-7]}}
  %t0 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %t1 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %t2 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %t3 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %t4 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %t5 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %t6 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %t7 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %t8 = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  %d0 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t0, x86_amx %t0, x86_amx %t0)
  %d1 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t1, x86_amx %t1, x86_amx %t1)
  %d2 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t2, x86_amx %t2, x86_amx %t2)
  %d3 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t3, x86_amx %t3, x86_amx %t3)
  %d4 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t4, x86_amx %t4, x86_amx %t4)
  %d5 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t5, x86_amx %t5, x86_amx %t5)
  %d6 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t6, x86_amx %t6, x86_amx %t6)
  %d7 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t7, x86_amx %t7, x86_amx %t7)
  %d8 = call x86_amx @llvm.x86.tdpbssd.internal(i16 8, i16 32, i16 32, x86_amx %t8, x86_amx %t8, x86_amx %t8)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d0)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d1)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d2)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d3)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d4)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d5)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d6)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d7)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %out, i64 32, x86_amx %d8)
  ret void
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.tdpbssd.internal(i16, i16, i16, x86_amx, x86_amx, x86_amx)
declare void @llvm.x86.tilestored64.internal(i16, i16, ptr, i64, x86_amx)
declare x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32>)
declare <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx)
