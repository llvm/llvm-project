; REQUIRES: riscv-registered-target
; RUN: opt -mtriple=riscv64 -mattr=+v -passes='print<cost-model>' -disable-output < %s 2>&1 | FileCheck %s

define <8 x i32> @shl_cost(<8 x i32> %a, <8 x i32> %b) {
; CHECK-LABEL: 'shl_cost'
; CHECK: Cost Model: Found an estimated cost of
; CHECK: shl <8 x i32>
  %r = shl <8 x i32> %a, %b
  ret <8 x i32> %r
}

define <8 x i32> @srl_cost(<8 x i32> %a, <8 x i32> %b) {
; CHECK-LABEL: 'srl_cost'
; CHECK: Cost Model: Found an estimated cost of
; CHECK: lshr <8 x i32>
  %r = lshr <8 x i32> %a, %b
  ret <8 x i32> %r
}

define <8 x i32> @sra_cost(<8 x i32> %a, <8 x i32> %b) {
; CHECK-LABEL: 'sra_cost'
; CHECK: Cost Model: Found an estimated cost of
; CHECK: ashr <8 x i32>
  %r = ashr <8 x i32> %a, %b
  ret <8 x i32> %r
}
