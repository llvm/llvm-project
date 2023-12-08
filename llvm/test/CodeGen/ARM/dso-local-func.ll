;; Check that we emit a $local alias for a dso_local function definition
; RUN: llc -mtriple=armv7-linux-gnueabi -relocation-model=static < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,STATIC
; RUN: llc -mtriple=armv7-linux-gnueabi -relocation-model=pic < %s \
; RUN:   | FileCheck %s --check-prefixes=CHECK,PIC

define dso_local ptr @dsolocal_func() nounwind {
; CHECK-LABEL: 	.globl	dsolocal_func
; CHECK-NEXT: 	.p2align	2
; CHECK-NEXT: 	.type	dsolocal_func,%function
; CHECK-NEXT: 	.code	32
; CHECK-NEXT: dsolocal_func:
; PIC-NEXT: .Ldsolocal_func$local:
; PIC-NEXT: .type .Ldsolocal_func$local,%function
; CHECK-NEXT: 	.fnstart
; CHECK-NEXT: @ %bb.0:
; STATIC-NEXT: 	movw	r0, :lower16:dsolocal_func
; STATIC-NEXT: 	movt	r0, :upper16:dsolocal_func
; STATIC-NEXT: 	bx	lr
; PIC-NEXT:     ldr	r0, .LCPI0_0
; PIC-NEXT:   .LPC0_0:
; PIC-NEXT:     add	r0, pc, r0
; PIC-NEXT:     bx	lr
; PIC-NEXT:     .p2align	2
; PIC-NEXT:   @ %bb.1:
; PIC-NEXT:   .LCPI0_0:
; PIC-NEXT:     .long	.Ldsolocal_func$local-(.LPC0_0+8)
; CHECK-NEXT: .Lfunc_end0:
; CHECK-NEXT: 	.size	dsolocal_func, .Lfunc_end0-dsolocal_func
; PIC-NEXT:     .size .Ldsolocal_func$local, .Lfunc_end0-dsolocal_func
; CHECK-NEXT: 	.cantunwind
; CHECK-NEXT: 	.fnend
  ret ptr @dsolocal_func
}
