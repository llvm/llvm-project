; RUN: llc < %s -march=x86

define i32 @test1() nounwind {
	; Dest is AX, dest type = i32.
        %tmp4 = call i32 asm sideeffect "FROB $0", "={ax}"()
        ret i32 %tmp4
}

define void @test2(i32 %V) nounwind {
	; input is AX, in type = i32.
        call void asm sideeffect "FROB $0", "{ax}"(i32 %V)
        ret void
}

define void @test3() nounwind {
        ; FP constant as a memory operand.
        tail call void asm sideeffect "frob $0", "m"( float 0x41E0000000000000)
        ret void
}

define void @test4() nounwind {
       ; J means a constant in range 0 to 63.
       tail call void asm sideeffect "bork $0", "J"(i32 37) nounwind
       ret void
}

; rdar://9738585
define i32 @test5() nounwind {
entry:
  %0 = tail call i32 asm "test", "=l,~{dirflag},~{fpsr},~{flags}"() nounwind
  ret i32 0
}

; rdar://9777108 PR10352
define void @test6(i1 zeroext %desired) nounwind {
entry:
  tail call void asm sideeffect "foo $0", "q,~{dirflag},~{fpsr},~{flags}"(i1 %desired) nounwind
  ret void
}

define void @test7(i1 zeroext %desired, i32* %p) nounwind {
entry:
  %0 = tail call i8 asm sideeffect "xchg $0, $1", "=r,*m,0,~{memory},~{dirflag},~{fpsr},~{flags}"(i32* %p, i1 %desired) nounwind
  ret void
}

; <rdar://problem/11542429>
; The constrained GR32_ABCD register class of the 'q' constraint requires
; special handling after the preceding outputs used up eax-edx.
define void @constrain_abcd(i8* %h) nounwind ssp {
entry:
  %0 = call { i32, i32, i32, i32, i32 } asm sideeffect "", "=&r,=&r,=&r,=&r,=&q,r,~{ecx},~{memory},~{dirflag},~{fpsr},~{flags}"(i8* %h) nounwind
  ret void
}
