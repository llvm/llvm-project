; RUN: llc < %s -march=x86-64 -mattr=+avx -no-integrated-as
; rdar://7066579

	%0 = type { i64, i64, i64, i64, i64 }		; type %0

define void @test1() nounwind {
entry:
	%asmtmp = call %0 asm sideeffect "mov    %cr0, $0       \0Amov    %cr2, $1       \0Amov    %cr3, $2       \0Amov    %cr4, $3       \0Amov    %cr8, $0       \0A", "=q,=q,=q,=q,=q,~{dirflag},~{fpsr},~{flags}"() nounwind		; <%0> [#uses=0]
	ret void
}

; PR9602
define void @test2(float %tmp) nounwind {
  call void asm sideeffect "$0", "q"(float %tmp) nounwind
  call void asm sideeffect "$0", "Q"(float %tmp) nounwind
  ret void
}

define void @test3(double %tmp) nounwind {
  call void asm sideeffect "$0", "q"(double %tmp) nounwind
  ret void
}

; rdar://10392864
define void @test4(i8 signext %val, i8 signext %a, i8 signext %b, i8 signext %c, i8 signext %d) nounwind {
entry:
  %0 = tail call { i8, i8, i8, i8, i8 } asm "foo $1, $2, $3, $4, $1\0Axchgb ${0:b}, ${0:h}", "=q,={ax},={bx},={cx},={dx},0,1,2,3,4,~{dirflag},~{fpsr},~{flags}"(i8 %val, i8 %a, i8 %b, i8 %c, i8 %d) nounwind
  ret void
}

; rdar://10614894
define <8 x float> @test5(<8 x float> %a, <8 x float> %b) nounwind {
entry:
  %0 = tail call <8 x float> asm "vperm2f128 $3, $2, $1, $0", "=x,x,x,i,~{dirflag},~{fpsr},~{flags}"(<8 x float> %a, <8 x float> %b, i32 16) nounwind
  ret <8 x float> %0
}

