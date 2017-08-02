; RUN: llc < %s -mtriple=x86_64-linux-gnu -regalloc=fast -optimize-regalloc=0 -relocation-model=pic | FileCheck %s
; PR4004

; CHECK: {{leaq.*TLSGD}}
; CHECK: {{__tls_get_addr}}

@i = thread_local global i32 15

define i32 @f() {
entry:
	%tmp1 = load i32, i32* @i
	ret i32 %tmp1
}
