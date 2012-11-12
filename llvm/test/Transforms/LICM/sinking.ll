; RUN: opt < %s -basicaa -licm -S | FileCheck %s

declare i32 @strlen(i8*) readonly

declare void @foo()

; Sink readonly function.
define i32 @test1(i8* %P) {
	br label %Loop

Loop:		; preds = %Loop, %0
	%A = call i32 @strlen( i8* %P ) readonly
	br i1 false, label %Loop, label %Out

Out:		; preds = %Loop
	ret i32 %A
; CHECK: @test1
; CHECK: Out:
; CHECK-NEXT: call i32 @strlen
; CHECK-NEXT: ret i32 %A
}

declare double @sin(double) readnone

; Sink readnone function out of loop with unknown memory behavior.
define double @test2(double %X) {
	br label %Loop

Loop:		; preds = %Loop, %0
	call void @foo( )
	%A = call double @sin( double %X ) readnone
	br i1 true, label %Loop, label %Out

Out:		; preds = %Loop
	ret double %A
; CHECK: @test2
; CHECK: Out:
; CHECK-NEXT: call double @sin
; CHECK-NEXT: ret double %A
}

; This testcase checks to make sure the sinker does not cause problems with
; critical edges.
define void @test3() {
Entry:
	br i1 false, label %Loop, label %Exit
Loop:
	%X = add i32 0, 1
	br i1 false, label %Loop, label %Exit
Exit:
	%Y = phi i32 [ 0, %Entry ], [ %X, %Loop ]
	ret void
        
; CHECK: @test3
; CHECK:     Exit.loopexit:
; CHECK-NEXT:  %X = add i32 0, 1
; CHECK-NEXT:  br label %Exit

}

; If the result of an instruction is only used outside of the loop, sink
; the instruction to the exit blocks instead of executing it on every
; iteration of the loop.
;
define i32 @test4(i32 %N) {
Entry:
	br label %Loop
Loop:		; preds = %Loop, %Entry
	%N_addr.0.pn = phi i32 [ %dec, %Loop ], [ %N, %Entry ]	
	%tmp.6 = mul i32 %N, %N_addr.0.pn		; <i32> [#uses=1]
	%tmp.7 = sub i32 %tmp.6, %N		; <i32> [#uses=1]
	%dec = add i32 %N_addr.0.pn, -1		; <i32> [#uses=1]
	%tmp.1 = icmp ne i32 %N_addr.0.pn, 1		; <i1> [#uses=1]
	br i1 %tmp.1, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %tmp.7
; CHECK: @test4
; CHECK:     Out:
; CHECK-NEXT:  mul i32 %N, %N_addr.0.pn
; CHECK-NEXT:  sub i32 %tmp.6, %N
; CHECK-NEXT:  ret i32
}

; To reduce register pressure, if a load is hoistable out of the loop, and the
; result of the load is only used outside of the loop, sink the load instead of
; hoisting it!
;
@X = global i32 5		; <i32*> [#uses=1]

define i32 @test5(i32 %N) {
Entry:
	br label %Loop
Loop:		; preds = %Loop, %Entry
	%N_addr.0.pn = phi i32 [ %dec, %Loop ], [ %N, %Entry ]	
	%tmp.6 = load i32* @X		; <i32> [#uses=1]
	%dec = add i32 %N_addr.0.pn, -1		; <i32> [#uses=1]
	%tmp.1 = icmp ne i32 %N_addr.0.pn, 1		; <i1> [#uses=1]
	br i1 %tmp.1, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %tmp.6
; CHECK: @test5
; CHECK:     Out:
; CHECK-NEXT:  %tmp.6 = load i32* @X
; CHECK-NEXT:  ret i32 %tmp.6
}



; The loop sinker was running from the bottom of the loop to the top, causing
; it to miss opportunities to sink instructions that depended on sinking other
; instructions from the loop.  Instead they got hoisted, which is better than
; leaving them in the loop, but increases register pressure pointlessly.

	%Ty = type { i32, i32 }
@X2 = external global %Ty

define i32 @test6() {
	br label %Loop
Loop:
	%dead = getelementptr %Ty* @X2, i64 0, i32 0
	%sunk2 = load i32* %dead
	br i1 false, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %sunk2
; CHECK: @test6
; CHECK:     Out:
; CHECK-NEXT:  %dead = getelementptr %Ty* @X2, i64 0, i32 0
; CHECK-NEXT:  %sunk2 = load i32* %dead
; CHECK-NEXT:  ret i32 %sunk2
}



; This testcase ensures that we can sink instructions from loops with
; multiple exits.
;
define i32 @test7(i32 %N, i1 %C) {
Entry:
	br label %Loop
Loop:		; preds = %ContLoop, %Entry
	%N_addr.0.pn = phi i32 [ %dec, %ContLoop ], [ %N, %Entry ]
	%tmp.6 = mul i32 %N, %N_addr.0.pn
	%tmp.7 = sub i32 %tmp.6, %N		; <i32> [#uses=2]
	%dec = add i32 %N_addr.0.pn, -1		; <i32> [#uses=1]
	br i1 %C, label %ContLoop, label %Out1
ContLoop:
	%tmp.1 = icmp ne i32 %N_addr.0.pn, 1
	br i1 %tmp.1, label %Loop, label %Out2
Out1:		; preds = %Loop
	ret i32 %tmp.7
Out2:		; preds = %ContLoop
	ret i32 %tmp.7
; CHECK: @test7
; CHECK:     Out1:
; CHECK-NEXT:  mul i32 %N, %N_addr.0.pn
; CHECK-NEXT:  sub i32 %tmp.6, %N
; CHECK-NEXT:  ret
; CHECK:     Out2:
; CHECK-NEXT:  mul i32 %N, %N_addr.0.pn
; CHECK-NEXT:  sub i32 %tmp.6
; CHECK-NEXT:  ret
}


; This testcase checks to make sure we can sink values which are only live on
; some exits out of the loop, and that we can do so without breaking dominator
; info.
define i32 @test8(i1 %C1, i1 %C2, i32* %P, i32* %Q) {
Entry:
	br label %Loop
Loop:		; preds = %Cont, %Entry
	br i1 %C1, label %Cont, label %exit1
Cont:		; preds = %Loop
	%X = load i32* %P		; <i32> [#uses=2]
	store i32 %X, i32* %Q
	%V = add i32 %X, 1		; <i32> [#uses=1]
	br i1 %C2, label %Loop, label %exit2
exit1:		; preds = %Loop
	ret i32 0
exit2:		; preds = %Cont
	ret i32 %V
; CHECK: @test8
; CHECK:     exit1:
; CHECK-NEXT:  ret i32 0
; CHECK:     exit2:
; CHECK-NEXT:  %V = add i32 %X, 1
; CHECK-NEXT:  ret i32 %V
}


define void @test9() {
loopentry.2.i:
	br i1 false, label %no_exit.1.i.preheader, label %loopentry.3.i.preheader
no_exit.1.i.preheader:		; preds = %loopentry.2.i
	br label %no_exit.1.i
no_exit.1.i:		; preds = %endif.8.i, %no_exit.1.i.preheader
	br i1 false, label %return.i, label %endif.8.i
endif.8.i:		; preds = %no_exit.1.i
	%inc.1.i = add i32 0, 1		; <i32> [#uses=1]
	br i1 false, label %no_exit.1.i, label %loopentry.3.i.preheader.loopexit
loopentry.3.i.preheader.loopexit:		; preds = %endif.8.i
	br label %loopentry.3.i.preheader
loopentry.3.i.preheader:		; preds = %loopentry.3.i.preheader.loopexit, %loopentry.2.i
	%arg_num.0.i.ph13000 = phi i32 [ 0, %loopentry.2.i ], [ %inc.1.i, %loopentry.3.i.preheader.loopexit ]		; <i32> [#uses=0]
	ret void
return.i:		; preds = %no_exit.1.i
	ret void

; CHECK: @test9
; CHECK: loopentry.3.i.preheader.loopexit:
; CHECK-NEXT:  %inc.1.i = add i32 0, 1
; CHECK-NEXT:  br label %loopentry.3.i.preheader
}


; Potentially trapping instructions may be sunk as long as they are guaranteed
; to be executed.
define i32 @test10(i32 %N) {
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
        
; CHECK: @test10
; CHECK: Out: 
; CHECK-NEXT:  %tmp.6 = sdiv i32 %N, %N_addr.0.pn
; CHECK-NEXT:  ret i32 %tmp.6
}

; Should delete, not sink, dead instructions.
define void @test11() {
	br label %Loop
Loop:
	%dead = getelementptr %Ty* @X2, i64 0, i32 0
	br i1 false, label %Loop, label %Out
Out:
	ret void
; CHECK: @test11
; CHECK:     Out:
; CHECK-NEXT:  ret void
}


