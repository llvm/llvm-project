; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+acev1 -O0 \
; RUN:   -verify-machineinstrs | FileCheck %s \
; RUN:   --implicit-check-not=tileloadd --implicit-check-not=tilestored

; A tile PHI becomes a load from a PHI address, so 16 row moves off it.

define void @phi_tile(i32 %cond, ptr %buf, <64 x i8> %a, <64 x i8> %b) nounwind {
; CHECK-LABEL: phi_tile:
; CHECK:       %join
; CHECK:         ldtilecfg
; CHECK:         vmovups (%r{{[a-z0-9]+}}), %zmm{{[0-9]+}}
; CHECK-NEXT:    tilemovrow $0, %zmm{{[0-9]+}}, %tmm{{[0-7]}}
; CHECK:         vmovups 960(%r{{[a-z0-9]+}}), %zmm{{[0-9]+}}
; CHECK-NEXT:    tilemovrow $15, %zmm{{[0-9]+}}, %tmm{{[0-7]}}
entry:
  %c = icmp ne i32 %cond, 0
  br i1 %c, label %bb1, label %bb2

bb1:
  %t1 = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  br label %join

bb2:
  %z = call x86_amx @llvm.x86.acev1.tilezero.internal(i16 16, i16 64)
  %t2 = call x86_amx @llvm.x86.acev1.top4buud.internal(i16 16, i16 64, x86_amx %z, <64 x i8> %a, <64 x i8> %b)
  br label %join

join:
  %t = phi x86_amx [ %t1, %bb1 ], [ %t2, %bb2 ]
  %row = call <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16 16, i16 64, x86_amx %t, i32 0)
  store <16 x i32> %row, ptr %buf, align 64
  ret void
}

declare x86_amx @llvm.x86.acev1.tilezero.internal(i16, i16)
declare x86_amx @llvm.x86.acev1.top4buud.internal(i16, i16, x86_amx, <64 x i8>, <64 x i8>)
declare <16 x i32> @llvm.x86.acev1.tilemovrowextract.internal(i16, i16, x86_amx, i32)
