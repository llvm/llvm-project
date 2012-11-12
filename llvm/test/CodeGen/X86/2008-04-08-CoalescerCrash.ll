; RUN: llc < %s -march=x86 -mattr=+mmx

define i32 @t2() nounwind  {
entry:
	tail call void asm sideeffect "# top of block", "~{dirflag},~{fpsr},~{flags},~{di},~{si},~{dx},~{cx},~{ax}"( ) nounwind 
	tail call void asm sideeffect ".file \224443946.c\22", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect ".line 8", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	%tmp1 = tail call x86_mmx asm sideeffect "movd $1, $0", "=={mm4},{bp},~{dirflag},~{fpsr},~{flags},~{memory}"( i32 undef ) nounwind 		; <x86_mmx> [#uses=1]
	tail call void asm sideeffect ".file \224443946.c\22", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect ".line 9", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	%tmp3 = tail call i32 asm sideeffect "movd $1, $0", "=={bp},{mm3},~{dirflag},~{fpsr},~{flags},~{memory}"( x86_mmx undef ) nounwind 		; <i32> [#uses=1]
	tail call void asm sideeffect ".file \224443946.c\22", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect ".line 10", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect "movntq $0, 0($1,$2)", "{mm0},{di},{bp},~{dirflag},~{fpsr},~{flags},~{memory}"( x86_mmx undef, i32 undef, i32 %tmp3 ) nounwind 
	tail call void asm sideeffect ".file \224443946.c\22", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect ".line 11", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	%tmp8 = tail call i32 asm sideeffect "movd $1, $0", "=={bp},{mm4},~{dirflag},~{fpsr},~{flags},~{memory}"( x86_mmx %tmp1 ) nounwind 		; <i32> [#uses=0]
	ret i32 undef
}
