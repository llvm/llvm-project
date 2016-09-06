; RUN: llc < %s -march=arm -mtriple=arm-linux-gnueabi | \
; RUN:     grep "__aeabi_read_tp"

define i8* @test() {
entry:
	%tmp1 = call i8* @llvm.thread.pointer( )		; <i8*> [#uses=0]
	ret i8* %tmp1
}

declare i8* @llvm.thread.pointer()
