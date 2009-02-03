; This testcase ensures that we can sink instructions from loops with
; multiple exits.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | \
; RUN:    %prcontext mul 1 | grep {Out\[12\]:}

define i32 @test(i32 %N, i1 %C) {
Entry:
	br label %Loop
Loop:		; preds = %ContLoop, %Entry
	%N_addr.0.pn = phi i32 [ %dec, %ContLoop ], [ %N, %Entry ]		; <i32> [#uses=3]
	%tmp.6 = mul i32 %N, %N_addr.0.pn		; <i32> [#uses=1]
	%tmp.7 = sub i32 %tmp.6, %N		; <i32> [#uses=2]
	%dec = add i32 %N_addr.0.pn, -1		; <i32> [#uses=1]
	br i1 %C, label %ContLoop, label %Out1
ContLoop:		; preds = %Loop
	%tmp.1 = icmp ne i32 %N_addr.0.pn, 1		; <i1> [#uses=1]
	br i1 %tmp.1, label %Loop, label %Out2
Out1:		; preds = %Loop
	ret i32 %tmp.7
Out2:		; preds = %ContLoop
	ret i32 %tmp.7
}

