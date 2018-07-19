; RUN: opt < %s -basicaa -licm -S | FileCheck %s

declare i32 @strlen(i8*) readonly nounwind

declare void @foo()

; Sink readonly function.
define i32 @test1(i8* %P) {
	br label %Loop

Loop:		; preds = %Loop, %0
	%A = call i32 @strlen( i8* %P ) readonly
	br i1 false, label %Loop, label %Out

Out:		; preds = %Loop
	ret i32 %A
; CHECK-LABEL: @test1(
; CHECK: Out:
; CHECK-NEXT: call i32 @strlen
; CHECK-NEXT: ret i32 %A
}

declare double @sin(double) readnone nounwind

; Sink readnone function out of loop with unknown memory behavior.
define double @test2(double %X) {
	br label %Loop

Loop:		; preds = %Loop, %0
	call void @foo( )
	%A = call double @sin( double %X ) readnone
	br i1 true, label %Loop, label %Out

Out:		; preds = %Loop
	ret double %A
; CHECK-LABEL: @test2(
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
        
; CHECK-LABEL: @test3(
; CHECK:     Exit.loopexit:
; CHECK-NEXT:  %X.le = add i32 0, 1
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
; CHECK-LABEL: @test4(
; CHECK:     Out:
; CHECK-NEXT:  %[[LCSSAPHI:.*]] = phi i32 [ %N_addr.0.pn
; CHECK-NEXT:  mul i32 %N, %[[LCSSAPHI]]
; CHECK-NEXT:  sub i32 %tmp.6.le, %N
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
	%tmp.6 = load i32, i32* @X		; <i32> [#uses=1]
	%dec = add i32 %N_addr.0.pn, -1		; <i32> [#uses=1]
	%tmp.1 = icmp ne i32 %N_addr.0.pn, 1		; <i1> [#uses=1]
	br i1 %tmp.1, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %tmp.6
; CHECK-LABEL: @test5(
; CHECK:     Out:
; CHECK-NEXT:  %tmp.6.le = load i32, i32* @X
; CHECK-NEXT:  ret i32 %tmp.6.le
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
	%dead = getelementptr %Ty, %Ty* @X2, i64 0, i32 0
	%sunk2 = load i32, i32* %dead
	br i1 false, label %Loop, label %Out
Out:		; preds = %Loop
	ret i32 %sunk2
; CHECK-LABEL: @test6(
; CHECK:     Out:
; CHECK-NEXT:  %dead.le = getelementptr %Ty, %Ty* @X2, i64 0, i32 0
; CHECK-NEXT:  %sunk2.le = load i32, i32* %dead.le
; CHECK-NEXT:  ret i32 %sunk2.le
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
; CHECK-LABEL: @test7(
; CHECK:     Out1:
; CHECK-NEXT:  %[[LCSSAPHI:.*]] = phi i32 [ %N_addr.0.pn
; CHECK-NEXT:  mul i32 %N, %[[LCSSAPHI]]
; CHECK-NEXT:  sub i32 %tmp.6.le, %N
; CHECK-NEXT:  ret
; CHECK:     Out2:
; CHECK-NEXT:  %[[LCSSAPHI:.*]] = phi i32 [ %N_addr.0.pn
; CHECK-NEXT:  mul i32 %N, %[[LCSSAPHI]]
; CHECK-NEXT:  sub i32 %tmp.6.le4, %N
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
	%X = load i32, i32* %P		; <i32> [#uses=2]
	store i32 %X, i32* %Q
	%V = add i32 %X, 1		; <i32> [#uses=1]
	br i1 %C2, label %Loop, label %exit2
exit1:		; preds = %Loop
	ret i32 0
exit2:		; preds = %Cont
	ret i32 %V
; CHECK-LABEL: @test8(
; CHECK:     exit1:
; CHECK-NEXT:  ret i32 0
; CHECK:     exit2:
; CHECK-NEXT:  %[[LCSSAPHI:.*]] = phi i32 [ %X
; CHECK-NEXT:  %V.le = add i32 %[[LCSSAPHI]], 1
; CHECK-NEXT:  ret i32 %V.le
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

; CHECK-LABEL: @test9(
; CHECK: loopentry.3.i.preheader.loopexit:
; CHECK-NEXT:  %inc.1.i.le = add i32 0, 1
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
        
; CHECK-LABEL: @test10(
; CHECK: Out: 
; CHECK-NEXT:  %[[LCSSAPHI:.*]] = phi i32 [ %N_addr.0.pn
; CHECK-NEXT:  %tmp.6.le = sdiv i32 %N, %[[LCSSAPHI]]
; CHECK-NEXT:  ret i32 %tmp.6.le
}

; Should delete, not sink, dead instructions.
define void @test11() {
	br label %Loop
Loop:
	%dead = getelementptr %Ty, %Ty* @X2, i64 0, i32 0
	br i1 false, label %Loop, label %Out
Out:
	ret void
; CHECK-LABEL: @test11(
; CHECK:     Out:
; CHECK-NEXT:  ret void
}

@c = common global [1 x i32] zeroinitializer, align 4

; Test a *many* way nested loop with multiple exit blocks both of which exit
; multiple loop nests. This exercises LCSSA corner cases.
define i32 @PR18753(i1* %a, i1* %b, i1* %c, i1* %d) {
entry:
  br label %l1.header

l1.header:
  %iv = phi i64 [ %iv.next, %l1.latch ], [ 0, %entry ]
  %arrayidx.i = getelementptr inbounds [1 x i32], [1 x i32]* @c, i64 0, i64 %iv
  br label %l2.header

l2.header:
  %x0 = load i1, i1* %c, align 4
  br i1 %x0, label %l1.latch, label %l3.preheader

l3.preheader:
  br label %l3.header

l3.header:
  %x1 = load i1, i1* %d, align 4
  br i1 %x1, label %l2.latch, label %l4.preheader

l4.preheader:
  br label %l4.header

l4.header:
  %x2 = load i1, i1* %a
  br i1 %x2, label %l3.latch, label %l4.body

l4.body:
  call void @f(i32* %arrayidx.i)
  %x3 = load i1, i1* %b
  %l = trunc i64 %iv to i32
  br i1 %x3, label %l4.latch, label %exit

l4.latch:
  call void @g()
  %x4 = load i1, i1* %b, align 4
  br i1 %x4, label %l4.header, label %exit

l3.latch:
  br label %l3.header

l2.latch:
  br label %l2.header

l1.latch:
  %iv.next = add nsw i64 %iv, 1
  br label %l1.header

exit:
  %lcssa = phi i32 [ %l, %l4.latch ], [ %l, %l4.body ]
; CHECK-LABEL: @PR18753(
; CHECK:       exit:
; CHECK-NEXT:    %[[LCSSAPHI:.*]] = phi i64 [ %iv, %l4.latch ], [ %iv, %l4.body ]
; CHECK-NEXT:    %l.le = trunc i64 %[[LCSSAPHI]] to i32
; CHECK-NEXT:    ret i32 %l.le

  ret i32 %lcssa
}

; Can't sink stores out of exit blocks containing indirectbr instructions
; because loop simplify does not create dedicated exits for such blocks. Test
; that by sinking the store from lab21 to lab22, but not further.
define void @test12() {
; CHECK-LABEL: @test12
  br label %lab4

lab4:
  br label %lab20

lab5:
  br label %lab20

lab6:
  br label %lab4

lab7:
  br i1 undef, label %lab8, label %lab13

lab8:
  br i1 undef, label %lab13, label %lab10

lab10:
  br label %lab7

lab13:
  ret void

lab20:
  br label %lab21

lab21:
; CHECK: lab21:
; CHECK-NOT: store
; CHECK: br i1 false, label %lab21, label %lab22
  store i32 36127957, i32* undef, align 4
  br i1 undef, label %lab21, label %lab22

lab22:
; CHECK: lab22:
; CHECK: store
; CHECK-NEXT: indirectbr i8* undef
  indirectbr i8* undef, [label %lab5, label %lab6, label %lab7]
}

; Test that we don't crash when trying to sink stores and there's no preheader
; available (which is used for creating loads that may be used by the SSA
; updater)
define void @test13() {
; CHECK-LABEL: @test13
  br label %lab59

lab19:
  br i1 undef, label %lab20, label %lab38

lab20:
  br label %lab60

lab21:
  br i1 undef, label %lab22, label %lab38

lab22:
  br label %lab38

lab38:
  ret void

lab59:
  indirectbr i8* undef, [label %lab60, label %lab38]

lab60:
; CHECK: lab60:
; CHECK: store
; CHECK-NEXT: indirectbr
  store i32 2145244101, i32* undef, align 4
  indirectbr i8* undef, [label %lab21, label %lab19]
}

; Check if LICM can sink a sinkable instruction the exit blocks through
; a non-trivially replacable PHI node.
;
; CHECK-LABEL: @test14
; CHECK-LABEL: Loop:
; CHECK-NOT: mul
; CHECK-NOT: sub
;
; CHECK-LABEL: Out12.split.loop.exit:
; CHECK: %[[LCSSAPHI:.*]] = phi i32 [ %N_addr.0.pn, %ContLoop ]
; CHECK: %[[MUL:.*]] = mul i32 %N, %[[LCSSAPHI]]
; CHECK: br label %Out12
;
; CHECK-LABEL: Out12.split.loop.exit1:
; CHECK: %[[LCSSAPHI2:.*]] = phi i32 [ %N_addr.0.pn, %Loop ]
; CHECK: %[[MUL2:.*]] = mul i32 %N, %[[LCSSAPHI2]]
; CHECK: %[[SUB:.*]] = sub i32 %[[MUL2]], %N
; CHECK: br label %Out12
;
; CHECK-LABEL: Out12:
; CHECK: phi i32 [ %[[MUL]], %Out12.split.loop.exit ], [ %[[SUB]], %Out12.split.loop.exit1 ]
define i32 @test14(i32 %N, i32 %N2, i1 %C) {
Entry:
        br label %Loop
Loop:
        %N_addr.0.pn = phi i32 [ %dec, %ContLoop ], [ %N, %Entry ]
        %sink.mul = mul i32 %N, %N_addr.0.pn
        %sink.sub = sub i32 %sink.mul, %N
        %dec = add i32 %N_addr.0.pn, -1
        br i1 %C, label %ContLoop, label %Out12
ContLoop:
        %tmp.1 = icmp ne i32 %N_addr.0.pn, 1
        br i1 %tmp.1, label %Loop, label %Out12
Out12:
  %tmp = phi i32 [%sink.mul,  %ContLoop], [%sink.sub, %Loop]
  ret i32 %tmp
}

; In this test, splitting predecessors is not really required because the
; operations of sinkable instructions (sub and mul) are same. In this case, we
; can sink the same sinkable operations and modify the PHI to pass the operands
; to the shared operations. As of now, we split predecessors of non-trivially
; replicalbe PHIs by default in LICM because all incoming edges of a
; non-trivially replacable PHI in LCSSA is critical.
;
; CHECK-LABEL: @test15
; CHECK-LABEL: Loop:
; CHECK-NOT: mul
; CHECK-NOT: sub
;
; CHECK-LABEL: Out12.split.loop.exit:
; CHECK: %[[LCSSAPHI:.*]] = phi i32 [ %N_addr.0.pn, %ContLoop ]
; CHECK: %[[MUL:.*]] = mul i32 %N, %[[LCSSAPHI]]
; CHECK: %[[SUB:.*]] = sub i32 %[[MUL]], %N2
; CHECK: br label %Out12
;
; CHECK-LABEL: Out12.split.loop.exit1:
; CHECK: %[[LCSSAPHI2:.*]] = phi i32 [ %N_addr.0.pn, %Loop ]
; CHECK: %[[MUL2:.*]] = mul i32 %N, %[[LCSSAPHI2]]
; CHECK: %[[SUB2:.*]] = sub i32 %[[MUL2]], %N
; CHECK: br label %Out12
;
; CHECK-LABEL: Out12:
; CHECK: phi i32 [ %[[SUB]], %Out12.split.loop.exit ], [ %[[SUB2]], %Out12.split.loop.exit1 ]
define i32 @test15(i32 %N, i32 %N2, i1 %C) {
Entry:
        br label %Loop
Loop:
        %N_addr.0.pn = phi i32 [ %dec, %ContLoop ], [ %N, %Entry ]
        %sink.mul = mul i32 %N, %N_addr.0.pn
        %sink.sub = sub i32 %sink.mul, %N
        %sink.sub2 = sub i32 %sink.mul, %N2
        %dec = add i32 %N_addr.0.pn, -1
        br i1 %C, label %ContLoop, label %Out12
ContLoop:
        %tmp.1 = icmp ne i32 %N_addr.0.pn, 1
        br i1 %tmp.1, label %Loop, label %Out12
Out12:
  %tmp = phi i32 [%sink.sub2, %ContLoop], [%sink.sub, %Loop]
  ret i32 %tmp
}

; Sink through a non-trivially replacable PHI node which use the same sinkable
; instruction multiple times.
;
; CHECK-LABEL: @test16
; CHECK-LABEL: Loop:
; CHECK-NOT: mul
;
; CHECK-LABEL: Out.split.loop.exit:
; CHECK: %[[PHI:.*]] = phi i32 [ %l2, %ContLoop ]
; CHECK: br label %Out
;
; CHECK-LABEL: Out.split.loop.exit1:
; CHECK: %[[SINKABLE:.*]] = mul i32 %l2.lcssa, %t.le
; CHECK: br label %Out
;
; CHECK-LABEL: Out:
; CHECK: %idx = phi i32 [ %[[PHI]], %Out.split.loop.exit ], [ %[[SINKABLE]], %Out.split.loop.exit1 ]
define i32 @test16(i1 %c, i8** %P, i32* %P2, i64 %V) {
entry:
  br label %loop.ph
loop.ph:
  br label %Loop
Loop:
  %iv = phi i64 [ 0, %loop.ph ], [ %next, %ContLoop ]
  %l2 = call i32 @getv()
  %t = trunc i64 %iv to i32
  %sinkable = mul i32 %l2,  %t
  switch i32 %l2, label %ContLoop [
    i32 32, label %Out
    i32 46, label %Out
    i32 95, label %Out
  ]
ContLoop:
  %next = add nuw i64 %iv, 1
  %c1 = call i1 @getc()
  br i1 %c1, label %Loop, label %Out
Out:
  %idx = phi i32 [ %l2, %ContLoop ], [ %sinkable, %Loop ], [ %sinkable, %Loop ], [ %sinkable, %Loop ]
  ret i32 %idx
}

; Sink a sinkable instruction through multiple non-trivially replacable PHIs in
; differect exit blocks.
;
; CHECK-LABEL: @test17
; CHECK-LABEL: Loop:
; CHECK-NOT: mul
;
; CHECK-LABEL:OutA.split.loop.exit{{.*}}:
; CHECK:  %[[OP1:.*]] = phi i32 [ %N_addr.0.pn, %ContLoop1 ]
; CHECK:  %[[SINKABLE:.*]] = mul i32 %N, %[[OP1]]
; CHECK:  br label %OutA
;
; CHECK-LABEL:OutA:
; CHECK: phi i32{{.*}}[ %[[SINKABLE]], %OutA.split.loop.exit{{.*}} ]
;
; CHECK-LABEL:OutB.split.loop.exit{{.*}}:
; CHECK:  %[[OP2:.*]] = phi i32 [ %N_addr.0.pn, %ContLoop2 ]
; CHECK:  %[[SINKABLE2:.*]] = mul i32 %N, %[[OP2]]
; CHECK:  br label %OutB
;
; CHECK-LABEL:OutB:
; CHECK:  phi i32 {{.*}}[ %[[SINKABLE2]], %OutB.split.loop.exit{{.*}} ]
define i32 @test17(i32 %N, i32 %N2) {
Entry:
        br label %Loop
Loop:
        %N_addr.0.pn = phi i32 [ %dec, %ContLoop3 ], [ %N, %Entry ]
        %sink.mul = mul i32 %N, %N_addr.0.pn
        %c0 = call i1 @getc()
        br i1 %c0 , label %ContLoop1, label %OutA
ContLoop1:
        %c1 = call i1 @getc()
        br i1 %c1, label %ContLoop2, label %OutA

ContLoop2:
        %c2 = call i1 @getc()
        br i1 %c2, label %ContLoop3, label %OutB
ContLoop3:
        %c3 = call i1 @getc()
        %dec = add i32 %N_addr.0.pn, -1
        br i1 %c3, label %Loop, label %OutB
OutA:
        %tmp1 = phi i32 [%sink.mul, %ContLoop1], [%N2, %Loop]
        br label %Out12
OutB:
        %tmp2 = phi i32 [%sink.mul, %ContLoop2], [%dec, %ContLoop3]
        br label %Out12
Out12:
  %tmp = phi i32 [%tmp1, %OutA], [%tmp2, %OutB]
  ret i32 %tmp
}


; Sink a sinkable instruction through both trivially and non-trivially replacable PHIs.
;
; CHECK-LABEL: @test18
; CHECK-LABEL: Loop:
; CHECK-NOT: mul
; CHECK-NOT: sub
;
; CHECK-LABEL:Out12.split.loop.exit:
; CHECK:  %[[OP:.*]] = phi i32 [ %iv, %ContLoop ]
; CHECK:  %[[DEC:.*]] = phi i32 [ %dec, %ContLoop ]
; CHECK:  %[[SINKMUL:.*]] = mul i32 %N, %[[OP]]
; CHECK:  %[[SINKSUB:.*]] = sub i32 %[[SINKMUL]], %N2
; CHECK:  br label %Out12
;
; CHECK-LABEL:Out12.split.loop.exit1:
; CHECK:  %[[OP2:.*]] = phi i32 [ %iv, %Loop ]
; CHECK:  %[[SINKMUL2:.*]] = mul i32 %N, %[[OP2]]
; CHECK:  %[[SINKSUB2:.*]] = sub i32 %[[SINKMUL2]], %N2
; CHECK:  br label %Out12
;
; CHECK-LABEL:Out12:
; CHECK:  %tmp1 = phi i32 [ %[[SINKSUB]], %Out12.split.loop.exit ], [ %[[SINKSUB2]], %Out12.split.loop.exit1 ]
; CHECK:  %tmp2 = phi i32 [ %[[DEC]], %Out12.split.loop.exit ], [ %[[SINKSUB2]], %Out12.split.loop.exit1 ]
; CHECK:  %add = add i32 %tmp1, %tmp2
define i32 @test18(i32 %N, i32 %N2) {
Entry:
        br label %Loop
Loop:
        %iv = phi i32 [ %dec, %ContLoop ], [ %N, %Entry ]
        %sink.mul = mul i32 %N, %iv
        %sink.sub = sub i32 %sink.mul, %N2
        %c0 = call i1 @getc()
        br i1 %c0, label %ContLoop, label %Out12
ContLoop:
        %dec = add i32 %iv, -1
        %c1 = call i1 @getc()
        br i1 %c1, label %Loop, label %Out12
Out12:
  %tmp1 = phi i32 [%sink.sub, %ContLoop], [%sink.sub, %Loop]
  %tmp2 = phi i32 [%dec, %ContLoop], [%sink.sub, %Loop]
  %add = add i32 %tmp1, %tmp2
  ret i32 %add
}

; Do not sink an instruction through a non-trivially replacable PHI, to avoid
; assert while splitting predecessors, if the terminator of predecessor is an
; indirectbr.
; CHECK-LABEL: @test19
; CHECK-LABEL: L0:
; CHECK: %sinkable = mul
; CHECK: %sinkable2 = add

define i32 @test19(i1 %cond, i1 %cond2, i8* %address, i32 %v1) nounwind {
entry:
  br label %L0
L0:
  %indirect.goto.dest = select i1 %cond, i8* blockaddress(@test19, %exit), i8* %address
  %v2 = call i32 @getv()
  %sinkable = mul i32 %v1, %v2
  %sinkable2 = add i32 %v1, %v2
  indirectbr i8* %indirect.goto.dest, [label %L1, label %exit]

L1:
  %indirect.goto.dest2 = select i1 %cond2, i8* blockaddress(@test19, %exit), i8* %address
  indirectbr i8* %indirect.goto.dest2, [label %L0, label %exit]

exit:
  %r = phi i32 [%sinkable, %L0], [%sinkable2, %L1]
  ret i32 %r
}


; Do not sink through a non-trivially replacable PHI if splitting predecessors
; not allowed in SplitBlockPredecessors().
;
; CHECK-LABEL: @test20
; CHECK-LABEL: while.cond
; CHECK: %sinkable = mul
; CHECK: %sinkable2 = add
define void @test20(i32* %s, i1 %b, i32 %v1, i32 %v2) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %while.cond
while.cond:
  %v = call i32 @getv()
  %sinkable = mul i32 %v, %v2
  %sinkable2 = add  i32 %v, %v2
  br i1 %b, label %try.cont, label %while.body
while.body:
  invoke void @may_throw()
          to label %while.body2 unwind label %catch.dispatch
while.body2:
  invoke void @may_throw2()
          to label %while.cond unwind label %catch.dispatch
catch.dispatch:
  %.lcssa1 = phi i32 [ %sinkable, %while.body ], [ %sinkable2, %while.body2 ]
  %cp = cleanuppad within none []
  store i32 %.lcssa1, i32* %s
  cleanupret from %cp unwind to caller
try.cont:
  ret void
}

; The sinkable call should be sunk into an exit block split. After splitting
; the exit block, BlockColor for new blocks should be added properly so
; that we should be able to access valid ColorVector.
;
; CHECK-LABEL:@test21_pr36184
; CHECK-LABEL: Loop
; CHECK-NOT: %sinkableCall
; CHECK-LABEL:Out.split.loop.exit
; CHECK: %sinkableCall
define i32 @test21_pr36184(i8* %P) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %loop.ph

loop.ph:
  br label %Loop

Loop:
  %sinkableCall = call i32 @strlen( i8* %P ) readonly
  br i1 undef, label %ContLoop, label %Out

ContLoop:
  br i1 undef, label %Loop, label %Out

Out:
  %idx = phi i32 [ %sinkableCall, %Loop ], [0, %ContLoop ]
  ret i32 %idx
}

; We do not support splitting a landingpad block if BlockColors is not empty.
; CHECK-LABEL: @test22
; CHECK-LABEL: while.body2
; CHECK-LABEL: %mul
; CHECK-NOT: lpadBB.split{{.*}}
define void @test22(i1 %b, i32 %v1, i32 %v2) personality i32 (...)* @__CxxFrameHandler3 {
entry:
  br label %while.cond
while.cond:
  br i1 %b, label %try.cont, label %while.body

while.body:
  invoke void @may_throw()
          to label %while.body2 unwind label %lpadBB

while.body2:
  %v = call i32 @getv()
  %mul = mul i32 %v, %v2
  invoke void @may_throw2()
          to label %while.cond unwind label %lpadBB
lpadBB:
  %.lcssa1 = phi i32 [ 0, %while.body ], [ %mul, %while.body2 ]
  landingpad { i8*, i32 }
               catch i8* null
  br label %lpadBBSucc1

lpadBBSucc1:
  ret void

try.cont:
  ret void
}

declare void @may_throw()
declare void @may_throw2()
declare i32 @__CxxFrameHandler3(...)
declare i32 @getv()
declare i1 @getc()
declare void @f(i32*)
declare void @g()
