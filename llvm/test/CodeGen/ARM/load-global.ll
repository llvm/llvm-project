; RUN: llc < %s -mtriple=armv6-apple-darwin -relocation-model=static | FileCheck %s -check-prefix=STATIC
; RUN: llc < %s -mtriple=armv6-apple-darwin -relocation-model=dynamic-no-pic | FileCheck %s -check-prefix=DYNAMIC
; RUN: llc < %s -mtriple=armv6-apple-darwin -relocation-model=pic | FileCheck %s -check-prefix=PIC
; RUN: llc < %s -mtriple=thumbv6-apple-darwin -relocation-model=pic | FileCheck %s -check-prefix=PIC_T
; RUN: llc < %s -mtriple=armv7-apple-darwin -relocation-model=pic | FileCheck %s -check-prefix=PIC_V7
; RUN: llc < %s -mtriple=armv6-linux-gnueabi -relocation-model=pic | FileCheck %s -check-prefix=LINUX
; RUN: llc < %s -mtriple=thumbv6-linux-gnueabi -relocation-model=pic | FileCheck %s -check-prefix=LINUX_T

@G = external global i32

define i32 @test1() {
; STATIC: _test1:
; STATIC: ldr r0, LCPI0_0
; STATIC: ldr r0, [r0]
; STATIC: .long _G

; DYNAMIC: _test1:
; DYNAMIC: ldr r0, LCPI0_0
; DYNAMIC: ldr r0, [r0]
; DYNAMIC: ldr r0, [r0]
; DYNAMIC: .long L_G$non_lazy_ptr

; PIC: _test1
; PIC: ldr r0, LCPI0_0
; PIC: ldr r0, [pc, r0]
; PIC: ldr r0, [r0]
; PIC: .long L_G$non_lazy_ptr-(LPC0_0+8)

; PIC_T: _test1
; PIC_T: ldr r0, LCPI0_0
; PIC_T: add r0, pc
; PIC_T: ldr r0, [r0]
; PIC_T: ldr r0, [r0]
; PIC_T: .long L_G$non_lazy_ptr-(LPC0_0+4)

; PIC_V7: _test1
; PIC_V7: movw r0, :lower16:(L_G$non_lazy_ptr-(LPC0_0+8))
; PIC_V7: movt r0, :upper16:(L_G$non_lazy_ptr-(LPC0_0+8))
; PIC_V7: ldr r0, [pc, r0]
; PIC_V7: ldr r0, [r0]

; LINUX: test1
; LINUX: ldr r0, .LCPI0_0
; LINUX: ldr r0, [pc, r0]
; LINUX: ldr r0, [r0]
; LINUX: .long G(GOT_PREL)-((.LPC0_0+8)-.Ltmp0)

; LINUX_T: ldr r0, .LCPI0_0
; LINUX_T: add r0, pc
; LINUX_T: ldr r0, [r0]
; LINUX_T: ldr r0, [r0]
	%tmp = load i32, ptr @G
	ret i32 %tmp
}
