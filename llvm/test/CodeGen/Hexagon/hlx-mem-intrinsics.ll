; RUN: llc -march=hexagon -mcpu=hexagonv85 -mattr=+hlx < %s | FileCheck %s

; Verify that the HLX load/store memory intrinsics select through the
; LX_L16b_pseudo / LX_S16b_pseudo pseudos and are expanded (after
; packetization) into the real .tmp load and .new store instructions,
; each bundled with an LX_assign.

declare <1024 x i16> @llvm.hexagon.LX.L16b.tmp(ptr)
declare void @llvm.hexagon.LX.S16b.new(ptr, <1024 x i16>)
declare <1024 x i16> @llvm.hexagon.LX.abs.uh(<1024 x i16>)

; CHECK-LABEL: t:
; CHECK: {
; CHECK-DAG: L0.h = L0.h
; CHECK-DAG: L0.tmp = mem(r0)
; CHECK: }
; CHECK: L0.uh = abs(L0.h)
; CHECK: {
; CHECK-DAG: L0.h = L0.h
; CHECK-DAG: mem(r1) = L0.new
; CHECK: }
define void @t(ptr %p, ptr %q) {
  %a = call <1024 x i16> @llvm.hexagon.LX.L16b.tmp(ptr %p)
  %r = call <1024 x i16> @llvm.hexagon.LX.abs.uh(<1024 x i16> %a)
  call void @llvm.hexagon.LX.S16b.new(ptr %q, <1024 x i16> %r)
  ret void
}
