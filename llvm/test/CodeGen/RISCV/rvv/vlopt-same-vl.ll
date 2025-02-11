; RUN: llc -mtriple=riscv64 -mattr=+v -riscv-enable-vl-optimizer \
; RUN:   -verify-machineinstrs -debug-only=riscv-vl-optimizer -o - 2>&1 %s | FileCheck %s 

; REQUIRES: asserts

; GitHub Issue #123862 provided a case where the riscv-vl-optimizer pass was
; very slow. It was found that that case benefited greatly from aborting due
; to CommonVL == VLOp. Adding the case provided in the issue would show up
; as a long running test instead of a test failure. We would likley have a hard
; time figuring if that case had a regression. So instead, we check this output
; which was responsible for speeding it up. 

define <vscale x 4 x i32> @same_vl_imm(<vscale x 4 x i32> %passthru, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b) {
  ; CHECK: User VL is: 4
  ; CHECK: Abort due to CommonVL == VLOp, no point in reducing.
  %v = call <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(<vscale x 4 x i32> poison, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, i64 4)
  %w = call <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(<vscale x 4 x i32> poison, <vscale x 4 x i32> %v, <vscale x 4 x i32> %a, i64 4)
  ret <vscale x 4 x i32> %w
}

define <vscale x 4 x i32> @same_vl_reg(<vscale x 4 x i32> %passthru, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, i64 %vl) {
  ; CHECK: User VL is: %3:gprnox0
  ; CHECK: Abort due to CommonVL == VLOp, no point in reducing.
  %v = call <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(<vscale x 4 x i32> poison, <vscale x 4 x i32> %a, <vscale x 4 x i32> %b, i64 %vl)
  %w = call <vscale x 4 x i32> @llvm.riscv.vadd.nxv4i32.nxv4i32(<vscale x 4 x i32> poison, <vscale x 4 x i32> %v, <vscale x 4 x i32> %a, i64 %vl)
  ret <vscale x 4 x i32> %w
}
