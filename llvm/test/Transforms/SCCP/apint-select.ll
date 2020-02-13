; RUN: opt < %s -sccp -S | not grep select

@A = constant i32 10

define i712 @test1() {
        %P = getelementptr i32, i32* @A, i32 0
        %B = ptrtoint i32* %P to i64
        %BB = and i64 %B, undef
        %C = icmp sge i64 %BB, 0
	%X = select i1 %C, i712 0, i712 1
	ret i712 %X
}



define i712 @test2(i1 %C) {
	%X = select i1 %C, i712 0, i712 undef
	ret i712 %X
}


