; RUN: llc < %s -mtriple=thumbv6m-arm-none-eabi -relocation-model=static | FileCheck %s -check-prefixes=CHECK

define void @fn() #0 {
entry:
; CHECK-LABEL: fn:
; CHECK:       ldr [[REG:r[0-9]+]], .LCPI0_0
; CHECK-NEXT:  blx [[REG]]
; CHECK:       .LCPI0_0:
; CHECK-NEXT:  .long   bar
  call void @bar()
  ret void
}

define void @execute_only_fn() #1 {
; CHECK-LABEL: execute_only_fn:
; CHECK:       movs    [[REG0:r[0-9]+]], :upper8_15:bar
; CHECK-NEXT:  lsls    [[REG0]], r0, #8
; CHECK-NEXT:  adds    [[REG0]], :upper0_7:bar
; CHECK-NEXT:  lsls    [[REG0]], r0, #8
; CHECK-NEXT:  adds    [[REG0]], :lower8_15:bar
; CHECK-NEXT:  lsls    [[REG0]], r0, #8
; CHECK-NEXT:  adds    [[REG0]], :lower0_7:bar
; CHECK-NEXT:  blx     [[REG0]]
; CHECK-NOT:   .LCPI1_0:

entry:
  call void @bar()
  ret void
}

attributes #0 = { noinline optnone "target-features"="+thumb-mode,+long-calls" }
attributes #1 = { noinline optnone "target-features"="+execute-only,+thumb-mode,+long-calls" }

declare dso_local void @bar()
