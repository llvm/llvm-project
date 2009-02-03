; RUN: llvm-as < %s | llc -march=x86

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
