; REQUIRES: riscv-registered-target

; RUN: llc -mtriple=riscv64 -mattr=+v -O2 -verify-machineinstrs < %s | FileCheck %s

; 我們用 <8 x i32> 並讓位移量來自第二個向量參數，確保走 VV 形式
;（不是 vx/vi）。每個函式只做一次對應的 shift。

; CHECK-LABEL: shl_vv
; CHECK:       vsll.vv
define <8 x i32> @shl_vv(<8 x i32> %a, <8 x i32> %b) {
  %r = shl <8 x i32> %a, %b
  ret <8 x i32> %r
}

; CHECK-LABEL: srl_vv
; CHECK:       vsrl.vv
define <8 x i32> @srl_vv(<8 x i32> %a, <8 x i32> %b) {
  %r = lshr <8 x i32> %a, %b
  ret <8 x i32> %r
}

; CHECK-LABEL: sra_vv
; CHECK:       vsra.vv
define <8 x i32> @sra_vv(<8 x i32> %a, <8 x i32> %b) {
  %r = ashr <8 x i32> %a, %b
  ret <8 x i32> %r
}
