; RUN: llc -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -code-model=large -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK-LARGE %s

@addr = global i8* null

define void @test_blockaddress() {
; CHECK-LABEL: test_blockaddress:
  store volatile i8* blockaddress(@test_blockaddress, %block), i8** @addr
  %val = load volatile i8*, i8** @addr
  indirectbr i8* %val, [label %block]
; CHECK: adrp [[DEST_HI:x[0-9]+]], [[DEST_LBL:.Ltmp[0-9]+]]
; CHECK: add [[DEST:x[0-9]+]], [[DEST_HI]], {{#?}}:lo12:[[DEST_LBL]]
; CHECK: str [[DEST]],
; CHECK: ldr [[NEWDEST:x[0-9]+]]
; CHECK: br [[NEWDEST]]

; CHECK-LARGE: movz [[ADDR_REG:x[0-9]+]], #:abs_g3:[[DEST_LBL:.Ltmp[0-9]+]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g2_nc:[[DEST_LBL]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g1_nc:[[DEST_LBL]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g0_nc:[[DEST_LBL]]
; CHECK-LARGE: str [[ADDR_REG]],
; CHECK-LARGE: ldr [[NEWDEST:x[0-9]+]]
; CHECK-LARGE: br [[NEWDEST]]

block:
  ret void
}
