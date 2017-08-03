; PR2094
; RUN: llc < %s -march=x86-64 -no-integrated-as | grep movslq
; RUN: llc < %s -march=x86-64 -no-integrated-as | grep addps
; RUN: llc < %s -march=x86-64 -no-integrated-as | grep paddd
; RUN: llc < %s -march=x86-64 -no-integrated-as | not grep movq

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"
target triple = "x86_64-apple-darwin8"

define i32 @test1(i8* %v, i8* %blk2, i8* %blk1, i32 %stride, i32 %h) nounwind  {
	%tmp12 = sext i32 %stride to i64		; <i64> [#uses=1]
	%mrv = call {i32, i8*, i8*} asm sideeffect "$0 $1 $2 $3 $4 $5 $6",
         "=r,=r,=r,r,r,r,r"( i64 %tmp12, i32 %h, i8* %blk1, i8* %blk2 ) nounwind
        %tmp6 = extractvalue {i32, i8*, i8*} %mrv, 0
	%tmp7 = call i32 asm sideeffect "set $0",
             "=r,~{dirflag},~{fpsr},~{flags}"( ) nounwind
	ret i32 %tmp7
}

define <4 x float> @test2() nounwind {
	%mrv = call {<4 x float>, <4 x float>} asm "set $0, $1", "=x,=x"()
	%a = extractvalue {<4 x float>, <4 x float>} %mrv, 0
	%b = extractvalue {<4 x float>, <4 x float>} %mrv, 1
	%c = fadd <4 x float> %a, %b
	ret <4 x float> %c
}

define <4 x i32> @test3() nounwind {
	%mrv = call {<4 x i32>, <4 x i32>} asm "set $0, $1", "=x,=x"()
	%a = extractvalue {<4 x i32>, <4 x i32>} %mrv, 0
	%b = extractvalue {<4 x i32>, <4 x i32>} %mrv, 1
	%c = add <4 x i32> %a, %b
	ret <4 x i32> %c
}

