; RUN: llc < %s -mtriple=thumbv7em-arm-none-eabi -relocation-model=static | FileCheck %s -check-prefixes=CHECK,STATIC
; RUN: llc < %s -mtriple=thumbv7em-arm-none-eabi -relocation-model=rwpi | FileCheck %s -check-prefixes=CHECK,RWPI

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
; STATIC-LABEL: execute_only_fn:
; STATIC:       movw    [[REG0:r[0-9]+]], :lower16:bar
; STATIC-NEXT:  movt    [[REG0]], :upper16:bar
; STATIC-NEXT:  blx     [[REG0]]
; STATIC-NOT:   .LCPI0_0:

; RWPI-LABEL: execute_only_fn:
; RWPI:       movw    [[REG0:r[0-9]+]], :lower16:bar
; RWPI-NEXT:  movt    [[REG0]], :upper16:bar
; RWPI-NEXT:  blx     [[REG0]]
; RWPI-NOT:   .LCPI1_0:
entry:
  call void @bar()
  ret void
}

attributes #0 = { noinline optnone "target-features"="+thumb-mode,+long-calls" }
attributes #1 = { noinline optnone "target-features"="+execute-only,+thumb-mode,+long-calls" }

declare dso_local void @bar()
