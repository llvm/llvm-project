; RUN: llc < %s -march=x86 | grep "(%esp)" | count 2
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
; a - a should be found and removed, leaving refs to only L and P
define i8* @test(i8* %a, i8* %L, i8* %P) nounwind {
entry:
        %0 = ptrtoint i8* %a to i32
        %1 = sub i32 -2, %0
        %2 = ptrtoint i8* %P to i32
        %3 = sub i32 0, %2
        %4 = ptrtoint i8* %L to i32
        %5 = add i32 %4, %3
	%6 = add i32 %5, %1         	; <i32> [#uses=1]
	%7 = getelementptr i8* %a, i32 %6		; <i8*> [#uses=1]
	br label %return

return:		; preds = %bb3
	ret i8* %7
}
