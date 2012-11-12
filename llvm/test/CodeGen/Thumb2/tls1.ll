; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi | \
; RUN:     grep "i(tpoff)"
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi | \
; RUN:     grep "__aeabi_read_tp"
; RUN: llc < %s -mtriple=thumbv7-linux-gnueabi \
; RUN:     -relocation-model=pic | grep "__tls_get_addr"


@i = thread_local global i32 15		; <i32*> [#uses=2]

define i32 @f() {
entry:
	%tmp1 = load i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32* @g() {
entry:
	ret i32* @i
}
