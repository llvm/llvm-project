; RUN: llc -mtriple=arm-- --relocation-model=rwpi %s -o - | \
; RUN: FileCheck %s --check-prefixes=RWPI
; RUN: llc -mtriple=arm-- --relocation-model=ropi %s -o - | \
; RUN: FileCheck %s --check-prefixes=ROPI
; RUN: llc -mtriple=arm-- --relocation-model=pic %s -o - | \
; RUN: FileCheck %s --check-prefixes=PIC

; RWPI:        ldr     {{r[0-9]+}}, .LCPI0_0
; RWPI:        .LCPI0_0:
; RWPI-NEXT:           .long   __stack_chk_guard(sbrel)

; ROPI:        ldr     {{r[0-9]+}}, .LCPI0_0
; ROPI:        .LCPI0_0:
; ROPI-NEXT:           .long   __stack_chk_guard

; PIC:         ldr     {{r[0-9]+}}, .LCPI0_0
; PIC:         .LCPI0_0:
; PIC-NEXT:    .Ltmp0:
; PIC-NEXT:            .long   __stack_chk_guard(GOT_PREL)-((.LPC0_0+8)-.Ltmp0)

define dso_local i32 @foo(i32 %t) nounwind sspstrong {
entry:
  %vla = alloca i32, i32 %t
  %call = call i32 @baz(ptr %vla)
  ret i32 %call
}

declare dso_local i32 @baz(ptr)

!llvm.module.flags = !{!0}
!0 = !{i32 7, !"PIC Level", i32 2}
