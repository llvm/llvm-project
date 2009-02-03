; Potentially trapping instructions may be sunk as long as they are guaranteed
; to be executed.
;
; RUN: llvm-as < %s | opt -licm | llvm-dis | %prcontext div 1 | grep Out: 

define i32 @test(i32 %N) {
Entry:
	br label %Loop
Loop:		; preds = %Loop, %Entry
	%N_addr.0.pn = phi i32 [ %dec, %Loop ], [ %N, %Entry ]		; <i32> [#uses=3]
	%tmp.6 = sdiv i32 %N, %N_addr.0.pn		; <i32> [#uses=1]
	%dec = add i32 %N_addr.0.pn, -1		; <i32> [#uses=1]
	%tmp.1 = icmp ne i32 %N_addr.0.pn, 0		; <i1> [#uses=1]
	br i1 %tmp.1, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %tmp.6
}

