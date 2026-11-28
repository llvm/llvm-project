; Check that a vselect whose condition comes from an icmp on <1024 x i32>
; operands is compiled end to end. The compare result is extracted from the
; register pair with LX_lo, which is expanded to a copy after register
; allocation.

; RUN: llc -mtriple=hexagon -mcpu=hexagonv85 -mattr=+hlx -verify-machineinstrs \
; RUN:   < %s | FileCheck %s

; CHECK-LABEL: test_vselect_i32cond:
; CHECK: .w = cmp.gt(
; CHECK: .h = mux(

define void @test_vselect_i32cond(ptr %a, ptr %b, ptr %out) {
entry:
  %va = tail call <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr %a)
  %vb = load <1024 x i16>, ptr %b, align 2048
  %cond = icmp slt <1024 x i32> %va, zeroinitializer
  %sel = select <1024 x i1> %cond, <1024 x i16> splat (i16 -1), <1024 x i16> %vb
  store <1024 x i16> %sel, ptr %out, align 2048
  ret void
}

declare <1024 x i32> @llvm.hexagon.LX.LL16b.tmp(ptr)
