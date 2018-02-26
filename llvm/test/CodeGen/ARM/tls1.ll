; RUN: llc < %s -mtriple=arm-linux-gnueabi | FileCheck %s
; RUN: llc < %s -mtriple=arm-linux-gnueabi -relocation-model=pic | \
; RUN:   FileCheck %s --check-prefix=PIC


; CHECK: i(TPOFF)
; CHECK: __aeabi_read_tp

; PIC: __tls_get_addr

@i = thread_local global i32 15		; <i32*> [#uses=2]

define i32 @f() {
entry:
	%tmp1 = load i32, i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32* @g() {
entry:
	ret i32* @i
}
