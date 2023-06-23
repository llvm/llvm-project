; RUN: llc -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -code-model=large -mtriple=aarch64-none-linux-gnu -aarch64-enable-atomic-cfg-tidy=0 -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK-LARGE %s
; RUN: llc -code-model=tiny -mtriple=aarch64-none-elf -aarch64-enable-atomic-cfg-tidy=0 -verify-machineinstrs < %s | FileCheck --check-prefix=CHECK-TINY %s

@addr = global ptr null

define void @test_blockaddress() {
; CHECK-LABEL: test_blockaddress:
  store volatile ptr blockaddress(@test_blockaddress, %block), ptr @addr
  %val = load volatile ptr, ptr @addr
  indirectbr ptr %val, [label %block]
; CHECK: adrp [[DEST_HI:x[0-9]+]], [[DEST_LBL:.Ltmp[0-9]+]]
; CHECK: add [[DEST:x[0-9]+]], [[DEST_HI]], {{#?}}:lo12:[[DEST_LBL]]
; CHECK: str [[DEST]],
; CHECK: ldr [[NEWDEST:x[0-9]+]]
; CHECK: br [[NEWDEST]]

; CHECK-LARGE: movz [[ADDR_REG:x[0-9]+]], #:abs_g0_nc:[[DEST_LBL:.Ltmp[0-9]+]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g1_nc:[[DEST_LBL]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g2_nc:[[DEST_LBL]]
; CHECK-LARGE: movk [[ADDR_REG]], #:abs_g3:[[DEST_LBL]]
; CHECK-LARGE: str [[ADDR_REG]],
; CHECK-LARGE: ldr [[NEWDEST:x[0-9]+]]
; CHECK-LARGE: br [[NEWDEST]]

; CHECK-TINY: adr [[DEST:x[0-9]+]], {{.Ltmp[0-9]+}}
; CHECK-TINY: str [[DEST]],
; CHECK-TINY: ldr [[NEWDEST:x[0-9]+]]
; CHECK-TINY: br [[NEWDEST]]

block:
  ret void
}
