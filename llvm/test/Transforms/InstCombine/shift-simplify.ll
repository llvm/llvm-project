; RUN: llvm-as < %s | opt -instcombine | llvm-dis | \
; RUN:    egrep {shl|lshr|ashr} | count 3

define i32 @test0(i32 %A, i32 %B, i32 %C) {
	%X = shl i32 %A, %C
	%Y = shl i32 %B, %C
	%Z = and i32 %X, %Y
	ret i32 %Z
}

define i32 @test1(i32 %A, i32 %B, i32 %C) {
	%X = lshr i32 %A, %C
	%Y = lshr i32 %B, %C
	%Z = or i32 %X, %Y
	ret i32 %Z
}

define i32 @test2(i32 %A, i32 %B, i32 %C) {
	%X = ashr i32 %A, %C
	%Y = ashr i32 %B, %C
	%Z = xor i32 %X, %Y
	ret i32 %Z
}

define i1 @test3(i32 %X) {
        %tmp1 = shl i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
}

define i1 @test4(i32 %X) {
        %tmp1 = lshr i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
}

define i1 @test5(i32 %X) {
        %tmp1 = ashr i32 %X, 7
        %tmp2 = icmp slt i32 %tmp1, 0
        ret i1 %tmp2
}

