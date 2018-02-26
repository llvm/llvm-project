; RUN: llc < %s -mtriple=arm-linux-gnueabi \
; RUN:   | FileCheck %s -check-prefix=CHECK-NONPIC
; RUN: llc < %s -mtriple=arm-linux-gnueabi -relocation-model=pic \
; RUN:   | FileCheck %s -check-prefix=CHECK-PIC

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32 @f() {
; CHECK-NONPIC-LABEL: f:
; CHECK-NONPIC: ldr {{r.}}, [pc, {{r.}}]
; CHECK-NONPIC: i(GOTTPOFF)
; CHECK-PIC-LABEL: f:
; CHECK-PIC: __tls_get_addr
entry:
	%tmp1 = load i32, i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32* @g() {
; CHECK-NONPIC-LABEL: g:
; CHECK-NONPIC: ldr {{r.}}, [pc, {{r.}}]
; CHECK-NONPIC: i(GOTTPOFF)
; CHECK-PIC-LABEL: g:
; CHECK-PIC: __tls_get_addr
entry:
	ret i32* @i
}
