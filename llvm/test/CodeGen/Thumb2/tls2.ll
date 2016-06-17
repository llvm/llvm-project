; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi | FileCheck %s -check-prefix=CHECK-NOT-PIC
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi -relocation-model=pic | FileCheck %s -check-prefix=CHECK-PIC

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32 @f() {
entry:
; CHECK-NOT-PIC-LABEL: f:
; CHECK-NOT-PIC: add r0, pc
; CHECK-NOT-PIC: ldr r1, [r0]
; CHECK-NOT-PIC: i(GOTTPOFF)

; CHECK-PIC-LABEL: f:
; CHECK-PIC: bl __tls_get_addr
	%tmp1 = load i32, i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32* @g() {
entry:
; CHECK-NOT-PIC-LABEL: g:
; CHECK-NOT-PIC: add r0, pc
; CHECK-NOT-PIC: ldr r1, [r0]
; CHECK-NOT-PIC: i(GOTTPOFF)

; CHECK-PIC-LABEL: g:
; CHECK-PIC: bl __tls_get_addr
	ret i32* @i
}
