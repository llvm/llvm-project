; RUN: llc < %s -march=x86-64 -mtriple=x86_64-linux-gnu -regalloc=fast -optimize-regalloc=0 -relocation-model=pic > %t2
; RUN: grep "leaq.*TLSGD" %t2
; RUN: grep "__tls_get_addr" %t2
; PR4004

@i = thread_local global i32 15

define i32 @f() {
entry:
	%tmp1 = load i32* @i
	ret i32 %tmp1
}
