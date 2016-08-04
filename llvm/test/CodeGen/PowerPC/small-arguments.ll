; RUN: llc -verify-machineinstrs < %s -march=ppc32 | not grep "extsh\|rlwinm"

declare signext i16 @foo()  

define i32 @test1(i16 signext %X) {
	%Y = sext i16 %X to i32  ;; dead
	ret i32 %Y
}

define i32 @test2(i16 zeroext %X) {
	%Y = sext i16 %X to i32
	%Z = and i32 %Y, 65535      ;; dead
	ret i32 %Z
}

define void @test3() {
	%tmp.0 = call signext i16 @foo()             ;; no extsh!
	%tmp.1 = icmp slt i16 %tmp.0, 1234
	br i1 %tmp.1, label %then, label %UnifiedReturnBlock

then:	
	call i32 @test1(i16 signext 0)
	ret void
UnifiedReturnBlock:
	ret void
}

define i32 @test4(i16* %P) {
        %tmp.1 = load i16, i16* %P
        %tmp.2 = zext i16 %tmp.1 to i32
        %tmp.3 = and i32 %tmp.2, 255
        ret i32 %tmp.3
}

define i32 @test5(i16* %P) {
        %tmp.1 = load i16, i16* %P
        %tmp.2 = bitcast i16 %tmp.1 to i16
        %tmp.3 = zext i16 %tmp.2 to i32
        %tmp.4 = and i32 %tmp.3, 255
        ret i32 %tmp.4
}

define i32 @test6(i32* %P) {
        %tmp.1 = load i32, i32* %P
        %tmp.2 = and i32 %tmp.1, 255
        ret i32 %tmp.2
}

define zeroext i16 @test7(float %a)  {
        %tmp.1 = fptoui float %a to i16
        ret i16 %tmp.1
}
