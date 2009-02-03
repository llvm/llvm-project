; RUN: llvm-as < %s | opt -constprop -disable-output
; PR2529
define <4 x i32> @test1(i32 %argc, i8** %argv) {
entry:  
        %foo = vicmp slt <4 x i32> undef, <i32 14, i32 undef, i32 undef, i32 undef>
        ret <4 x i32> %foo
}

define <4 x i32> @test2(i32 %argc, i8** %argv) {
entry:  
        %foo = vicmp slt <4 x i32> <i32 undef, i32 undef, i32 undef, i32
undef>, <i32 undef, i32 undef, i32 undef, i32 undef>
        ret <4 x i32> %foo
}


define <4 x i32> @test3() {
       %foo = vfcmp ueq <4 x float> <float 0.0, float 0.0, float 0.0, float
undef>, <float 1.0, float 1.0, float 1.0, float undef>
	ret <4 x i32> %foo
}

define <4 x i32> @test4() {
   %foo = vfcmp ueq <4 x float> <float 0.0, float 0.0, float 0.0, float 0.0>, <float 1.0, float 1.0, float 1.0, float 0.0>

	ret <4 x i32> %foo
}

