; RUN: llvm-as < %s | llc -march=ppc32 -mcpu=g5 -o %t -f
; RUN: grep vrlw %t
; RUN: not grep spr %t
; RUN: not grep vrsave %t

define <4 x i32> @test_rol() {
        ret <4 x i32> < i32 -11534337, i32 -11534337, i32 -11534337, i32 -11534337 >
}

define <4 x i32> @test_arg(<4 x i32> %A, <4 x i32> %B) {
        %C = add <4 x i32> %A, %B               ; <<4 x i32>> [#uses=1]
        ret <4 x i32> %C
}

