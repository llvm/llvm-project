; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-tile,+amx-int8 \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+amx-tile,+amx-int8,+acev1 \
; RUN:   -verify-machineinstrs | FileCheck %s

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

define <256 x i32> @amx_tile_to_vector_on_ace_target() nounwind {
; CHECK-LABEL: amx_tile_to_vector_on_ace_target:
; CHECK:         movb $1, {{.*}}(%rsp)
; CHECK:         tilezero %tmm0
; CHECK:         tilestored %tmm0, (%rdi,%rsi)
  %t = call x86_amx @llvm.x86.tilezero.internal(i16 16, i16 64)
  %v = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %t)
  ret <256 x i32> %v
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare void @llvm.x86.tilestored64.internal(i16, i16, ptr, i64, x86_amx)
declare x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32>)
declare <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx)
