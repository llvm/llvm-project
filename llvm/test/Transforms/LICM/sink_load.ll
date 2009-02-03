; To reduce register pressure, if a load is hoistable out of the loop, and the
; result of the load is only used outside of the loop, sink the load instead of
; hoisting it!
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | %prcontext load 1 | grep Out: 

@X = global i32 5		; <i32*> [#uses=1]

define i32 @test(i32 %N) {
Entry:
	br label %Loop
Loop:		; preds = %Loop, %Entry
	%N_addr.0.pn = phi i32 [ %dec, %Loop ], [ %N, %Entry ]		; <i32> [#uses=2]
	%tmp.6 = load i32* @X		; <i32> [#uses=1]
	%dec = add i32 %N_addr.0.pn, -1		; <i32> [#uses=1]
	%tmp.1 = icmp ne i32 %N_addr.0.pn, 1		; <i1> [#uses=1]
	br i1 %tmp.1, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %tmp.6
}

