; PR850
; RUN: llc < %s -mtriple=i686-- -x86-asm-syntax=att -no-integrated-as | FileCheck %s

; CHECK: {{movl 4[(]%eax[)],%ebp}}
; CHECK: {{movl 0[(]%eax[)], %ebx}}

define i32 @foo(i32 %__s.i.i, i32 %tmp5.i.i, i32 %tmp6.i.i, i32 %tmp7.i.i, i32 %tmp8.i.i) {
	%tmp9.i.i = call i32 asm sideeffect "push %ebp\0Apush %ebx\0Amovl 4($2),%ebp\0Amovl 0($2), %ebx\0Amovl $1,%eax\0Aint  $$0x80\0Apop  %ebx\0Apop %ebp", "={ax},i,0,{cx},{dx},{si},{di}"( i32 192, i32 %__s.i.i, i32 %tmp5.i.i, i32 %tmp6.i.i, i32 %tmp7.i.i, i32 %tmp8.i.i )		; <i32> [#uses=1]
	ret i32 %tmp9.i.i
}

