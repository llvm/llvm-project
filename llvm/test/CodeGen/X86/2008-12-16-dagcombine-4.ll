; RUN: llvm-as < %s | llc -march=x86 | grep "(%esp)" | count 2
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin9.5"
; a - a should be found and removed, leaving refs to only L and P
define i32 @test(i32 %a, i32 %L, i32 %P) nounwind {
entry:
        %0 = sub i32 %a, %L
        %1 = add i32 %P, %0
	%2 = sub i32 %1, %a
	br label %return

return:		; preds = %bb3
	ret i32 %2
}
