; RUN: llc < %s -mtriple=i686-pc-linux -mattr=+mmx

define i32 @t2() nounwind  {
entry:
	tail call void asm sideeffect "# top of block", "~{dirflag},~{fpsr},~{flags},~{di},~{si},~{dx},~{cx},~{ax}"( ) nounwind 
	tail call void asm sideeffect ".file \224443946.c\22", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect ".line 8", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	%tmp1 = tail call <1 x i64> asm sideeffect "movd $1, $0", "=={mm4},{bp},~{dirflag},~{fpsr},~{flags},~{memory}"( i32 undef ) nounwind 		; <<1 x i64>> [#uses=1]
	tail call void asm sideeffect ".file \224443946.c\22", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect ".line 9", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	%tmp3 = tail call i32 asm sideeffect "movd $1, $0", "=={bp},{mm3},~{dirflag},~{fpsr},~{flags},~{memory}"( <1 x i64> undef ) nounwind 		; <i32> [#uses=1]
	tail call void asm sideeffect ".file \224443946.c\22", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect ".line 10", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect "movntq $0, 0($1,$2)", "{mm0},{di},{bp},~{dirflag},~{fpsr},~{flags},~{memory}"( <1 x i64> undef, i32 undef, i32 %tmp3 ) nounwind 
	tail call void asm sideeffect ".file \224443946.c\22", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	tail call void asm sideeffect ".line 11", "~{dirflag},~{fpsr},~{flags}"( ) nounwind 
	%tmp8 = tail call i32 asm sideeffect "movd $1, $0", "=={bp},{mm4},~{dirflag},~{fpsr},~{flags},~{memory}"( <1 x i64> %tmp1 ) nounwind 		; <i32> [#uses=0]
	ret i32 undef
}
