; RUN: llc < %s -mtriple=armv6-apple-darwin -relocation-model=static | FileCheck %s -check-prefix=DarwinStatic
; RUN: llc < %s -mtriple=armv6-apple-darwin -relocation-model=dynamic-no-pic | FileCheck %s -check-prefix=DarwinDynamic
; RUN: llc < %s -mtriple=armv6-apple-darwin -relocation-model=pic | FileCheck %s -check-prefix=DarwinPIC
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -relocation-model=pic | FileCheck %s -check-prefix=LinuxPIC

@G = external global i32

define i32 @test1() {
	%tmp = load i32, i32* @G
	ret i32 %tmp
}

; DarwinStatic: _test1:
; DarwinStatic: 	ldr r0, LCPI0_0
; DarwinStatic:	        ldr r0, [r0]
; DarwinStatic:	        bx lr

; DarwinStatic: 	.p2align	2
; DarwinStatic:	LCPI0_0:
; DarwinStatic: 	.long	{{_G$}}


; DarwinDynamic: _test1:
; DarwinDynamic: 	ldr r0, LCPI0_0
; DarwinDynamic:        ldr r0, [r0]
; DarwinDynamic:        ldr r0, [r0]
; DarwinDynamic:        bx lr

; DarwinDynamic: 	.p2align	2
; DarwinDynamic:	LCPI0_0:
; DarwinDynamic: 	.long	L_G$non_lazy_ptr

; DarwinDynamic: 	.section __DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
; DarwinDynamic:	.p2align	2
; DarwinDynamic: L_G$non_lazy_ptr:
; DarwinDynamic:	.indirect_symbol _G
; DarwinDynamic:	.long	0



; DarwinPIC: _test1:
; DarwinPIC: 	ldr r0, LCPI0_0
; DarwinPIC: LPC0_0:
; DarwinPIC:    ldr r0, [pc, r0]
; DarwinPIC:    ldr r0, [r0]
; DarwinPIC-NOT: ldr
; DarwinPIC:    bx lr

; DarwinPIC: 	.p2align	2
; DarwinPIC: LCPI0_0:
; DarwinPIC: 	.long	L_G$non_lazy_ptr-(LPC0_0+8)

; DarwinPIC: 	.section __DATA,__nl_symbol_ptr,non_lazy_symbol_pointers
; DarwinPIC:	.p2align	2
; DarwinPIC: L_G$non_lazy_ptr:
; DarwinPIC:	.indirect_symbol _G
; DarwinPIC:	.long	0



; LinuxPIC-LABEL: test1:
; LinuxPIC: 	ldr r0, .LCPI0_0
	
; LinuxPIC: .LPC0_0:
; LinuxPIC: 	ldr r0, [pc, r0]
; LinuxPIC: 	ldr r0, [r0]
; LinuxPIC: 	bx lr

; LinuxPIC: .p2align 2
; LinuxPIC: .LCPI0_0:
; LinuxPIC: .Ltmp0:
; LinuxPIC:     .long   G(GOT_PREL)-((.LPC0_0+8)-.Ltmp0)
