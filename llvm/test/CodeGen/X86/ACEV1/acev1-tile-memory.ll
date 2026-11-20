; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O2 \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefixes=CHECK,O2 \
; RUN:   --implicit-check-not=tileloadd --implicit-check-not=tilestored
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 \
; RUN:   -verify-machineinstrs | FileCheck %s --check-prefix=CHECK \
; RUN:   --implicit-check-not=tileloadd --implicit-check-not=tilestored

; With no TILELOADD or TILESTORED, a tile in memory moves a row at a time.

@G = global <256 x i32> zeroinitializer, align 64

define void @accumulate_in_memory(<64 x i8> %a, <64 x i8> %b) nounwind {
; CHECK-LABEL: accumulate_in_memory:
; CHECK:           ldtilecfg
; O2-COUNT-16:     vmovaps {{[0-9]*}}({{%r[a-z0-9]+}}), %zmm{{[0-9]+}}
; CHECK-COUNT-16:  tilemovrow ${{[0-9]+}}, %zmm{{[0-9]+}}, %tmm0
; CHECK:           top4buud
; CHECK-COUNT-16:  tilemovrow ${{[0-9]+}}, %tmm0, %zmm{{[0-9]+}}
; O2-COUNT-16:     vmovaps %zmm{{[0-9]+}}, {{[0-9]*}}({{%r[a-z0-9]+}})
; CHECK:           tilerelease
  %old = load <256 x i32>, ptr @G, align 64
  %tile = call x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32> %old)
  %acc = call x86_amx @llvm.x86.acev1.top4buud.internal(i16 16, i16 64,
                                                       x86_amx %tile,
                                                       <64 x i8> %a,
                                                       <64 x i8> %b)
  %new = call <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx %acc)
  store <256 x i32> %new, ptr @G, align 64
  ret void
}

declare x86_amx @llvm.x86.acev1.top4buud.internal(i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare x86_amx @llvm.x86.cast.vector.to.tile.v256i32(<256 x i32>)
declare <256 x i32> @llvm.x86.cast.tile.to.vector.v256i32(x86_amx)
