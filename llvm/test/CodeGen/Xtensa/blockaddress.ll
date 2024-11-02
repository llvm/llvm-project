; RUN: llc --mtriple=xtensa < %s | FileCheck %s

@addr = global ptr null

define void @test_blockaddress() {

  store volatile ptr blockaddress(@test_blockaddress, %block), ptr @addr
; CHECK:      .literal_position
; CHECK-NEXT: .literal .LCPI0_0, addr
; CHECK-NEXT: .literal .LCPI0_1, .Ltmp0
; CHECK-LABEL: test_blockaddress:
; CHECK:      # %bb.0:
; CHECK-NEXT: l32r a8, .LCPI0_0
; CHECK-NEXT: l32r a9, .LCPI0_1
; CHECK-NEXT: s32i a9, a8, 0
; CHECK-NEXT: l32i a8, a8, 0
; CHECK-NEXT: jx a8
; CHECK-NEXT: .Ltmp0:
; CHECK-NEXT: .LBB0_1:

  %val = load volatile ptr, ptr @addr
  indirectbr ptr %val, [label %block]

block:
  ret void
}
