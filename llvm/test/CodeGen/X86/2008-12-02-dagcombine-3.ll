; RUN: llc < %s | grep add | count 2
; RUN: llc < %s | grep sub | grep -v subsections | count 1
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
; this should be rearranged to have two +s and one -
define i32 @test(i8* %a, i8* %L, i8* %P) nounwind {
entry:
        %0 = ptrtoint i8* %P to i32
        %1 = sub i32 -2, %0
        %2 = ptrtoint i8* %L to i32
        %3 = ptrtoint i8* %a to i32
	%4 = sub i32 %2, %3         	; <i32> [#uses=1]
	%5 = add i32 %1, %4		; <i32> [#uses=1]
	br label %return

return:		; preds = %bb3
	ret i32 %5
}
