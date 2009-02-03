; If the result of an instruction is only used outside of the loop, sink
; the instruction to the exit blocks instead of executing it on every
; iteration of the loop.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | %prcontext mul 1 | grep Out: 

define i32 @test(i32 %N) {
Entry:
	br label %Loop
Loop:		; preds = %Loop, %Entry
	%N_addr.0.pn = phi i32 [ %dec, %Loop ], [ %N, %Entry ]		; <i32> [#uses=3]
	%tmp.6 = mul i32 %N, %N_addr.0.pn		; <i32> [#uses=1]
	%tmp.7 = sub i32 %tmp.6, %N		; <i32> [#uses=1]
	%dec = add i32 %N_addr.0.pn, -1		; <i32> [#uses=1]
	%tmp.1 = icmp ne i32 %N_addr.0.pn, 1		; <i1> [#uses=1]
	br i1 %tmp.1, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %tmp.7
}

