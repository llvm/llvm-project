; RUN: llvm-as < %s | opt -instcombine -disable-output

define <4 x i32> @test(<4 x i32> %A) {
    %B = xor <4 x i32> %A, < i32 -1, i32 -1, i32 -1, i32 -1 > 
    %C = and <4 x i32> %B, < i32 -1, i32 -1, i32 -1, i32 -1 >
    ret <4 x i32> %C
}
