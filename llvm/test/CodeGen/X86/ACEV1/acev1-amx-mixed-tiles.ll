; RUN: not llc < %s -mtriple=x86_64-unknown-unknown \
; RUN:   -mattr=+amx-tile,+amx-int8,+acev1 -o /dev/null 2>&1 | FileCheck %s
; RUN: not llc < %s -mtriple=x86_64-unknown-unknown -O0 \
; RUN:   -mattr=+amx-tile,+amx-int8,+acev1 -o /dev/null 2>&1 | FileCheck %s

; One palette at a time, so the compiler cannot manage both kinds of tile.

; CHECK: error: mixed_tiles: Failed to config tile register, AMX and ACE tiles need different configurations
define void @mixed_tiles(ptr %p) nounwind {
  %a = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %p, i64 32, x86_amx %a)
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %r = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %c, i32 0)
  store <16 x i32> %r, ptr %p, align 64
  ret void
}

; A hand-written config is as bad: it takes the palette from a managed tile.

; CHECK: error: managed_amx_direct_ace: Failed to config tile register, AMX and ACE tiles need different configurations
define void @managed_amx_direct_ace(ptr %p) nounwind {
  %a = call x86_amx @llvm.x86.tilezero.internal(i16 8, i16 32)
  call void @llvm.x86.tilestored64.internal(i16 8, i16 32, ptr %p, i64 32, x86_amx %a)
  call void @llvm.x86.acev1.tilezero(i8 1)
  ret void
}

; CHECK: error: managed_ace_direct_amx: Failed to config tile register, AMX and ACE tiles need different configurations
define void @managed_ace_direct_amx(ptr %p) nounwind {
  call void @llvm.x86.tilestored64(i8 1, ptr %p, i64 64)
  %c = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %r = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %c, i32 0)
  store <16 x i32> %r, ptr %p, align 64
  ret void
}

declare x86_amx @llvm.x86.tilezero.internal(i16, i16)
declare void @llvm.x86.tilestored64.internal(i16, i16, ptr, i64, x86_amx)
declare void @llvm.x86.tilestored64(i8, ptr, i64)
declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare void @llvm.x86.acev1.tilezero(i8)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16, i16, x86_amx, i32)
