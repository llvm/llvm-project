; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnueabi | \
; RUN:     grep {i(gottpoff)}
; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnueabi | \
; RUN:     grep {ldr r., \[pc, r.\]}
; RUN: llvm-as < %s | llc -march=arm -mtriple=arm-linux-gnueabi \
; RUN:     -relocation-model=pic | grep {__tls_get_addr}

@i = external thread_local global i32		; <i32*> [#uses=2]

define i32 @f() {
entry:
	%tmp1 = load i32* @i		; <i32> [#uses=1]
	ret i32 %tmp1
}

define i32* @g() {
entry:
	ret i32* @i
}
