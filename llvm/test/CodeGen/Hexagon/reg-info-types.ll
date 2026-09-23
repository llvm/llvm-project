; RUN: llc -mtriple=hexagon -mattr=+hvxv60,+hvx-length128b < %s | FileCheck %s

; Test coverage for HexagonRegisterInfo: exercise different register class
; type mappings including HvxWR (double vector) and HvxQR (predicate vector).

; CHECK-LABEL: test_hvx_wr:
; CHECK: vadd
define <64 x i32> @test_hvx_wr(<64 x i32> %a, <64 x i32> %b) {
entry:
  %add = add <64 x i32> %a, %b
  ret <64 x i32> %add
}

; CHECK-LABEL: test_hvx_qr:
; CHECK: vmax
define <32 x i32> @test_hvx_qr(<32 x i32> %a, <32 x i32> %b) {
entry:
  %cmp = icmp sgt <32 x i32> %a, %b
  %sel = select <32 x i1> %cmp, <32 x i32> %a, <32 x i32> %b
  ret <32 x i32> %sel
}

; CHECK-LABEL: test_pred_reg:
; CHECK: cmp
define i32 @test_pred_reg(i32 %a, i32 %b, i32 %c) {
entry:
  %cmp = icmp sgt i32 %a, %b
  %sel = select i1 %cmp, i32 %a, i32 %c
  ret i32 %sel
}

