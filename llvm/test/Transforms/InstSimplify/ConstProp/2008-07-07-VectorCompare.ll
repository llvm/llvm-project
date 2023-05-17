; RUN: opt < %s -passes=instsimplify -disable-output
; PR2529
define <4 x i1> @test1(i32 %argc, ptr %argv) {
entry:  
        %foo = icmp slt <4 x i32> undef, <i32 14, i32 undef, i32 undef, i32 undef>
        ret <4 x i1> %foo
}

define <4 x i1> @test2(i32 %argc, ptr %argv) {
entry:  
        %foo = icmp slt <4 x i32> <i32 undef, i32 undef, i32 undef, i32
undef>, <i32 undef, i32 undef, i32 undef, i32 undef>
        ret <4 x i1> %foo
}


define <4 x i1> @test3() {
       %foo = fcmp ueq <4 x float> <float 0.0, float 0.0, float 0.0, float
undef>, <float 1.0, float 1.0, float 1.0, float undef>
	ret <4 x i1> %foo
}

define <4 x i1> @test4() {
	%foo = fcmp ueq <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>, <float 1.0, float 1.0, float 1.0, float 0.0>

	ret <4 x i1> %foo
}

