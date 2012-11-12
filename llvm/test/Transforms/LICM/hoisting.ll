; RUN: opt < %s -licm -S | FileCheck %s

@X = global i32 0		; <i32*> [#uses=1]

declare void @foo()

; This testcase tests for a problem where LICM hoists 
; potentially trapping instructions when they are not guaranteed to execute.
define i32 @test1(i1 %c) {
; CHECK: @test1
	%A = load i32* @X		; <i32> [#uses=2]
	br label %Loop
Loop:		; preds = %LoopTail, %0
	call void @foo( )
	br i1 %c, label %LoopTail, label %IfUnEqual
        
IfUnEqual:		; preds = %Loop
; CHECK: IfUnEqual:
; CHECK-NEXT: sdiv i32 4, %A
	%B1 = sdiv i32 4, %A		; <i32> [#uses=1]
	br label %LoopTail
        
LoopTail:		; preds = %IfUnEqual, %Loop
	%B = phi i32 [ 0, %Loop ], [ %B1, %IfUnEqual ]		; <i32> [#uses=1]
	br i1 %c, label %Loop, label %Out
Out:		; preds = %LoopTail
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}


declare void @foo2(i32) nounwind


;; It is ok and desirable to hoist this potentially trapping instruction.
define i32 @test2(i1 %c) {
; CHECK: @test2
; CHECK-NEXT: load i32* @X
; CHECK-NEXT: %B = sdiv i32 4, %A
	%A = load i32* @X		; <i32> [#uses=2]
	br label %Loop
Loop:
        ;; Should have hoisted this div!
	%B = sdiv i32 4, %A		; <i32> [#uses=2]
	call void @foo2( i32 %B )
	br i1 %c, label %Loop, label %Out
Out:		; preds = %Loop
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}


; This loop invariant instruction should be constant folded, not hoisted.
define i32 @test3(i1 %c) {
; CHECK: define i32 @test3
; CHECK: call void @foo2(i32 6)
	%A = load i32* @X		; <i32> [#uses=2]
	br label %Loop
Loop:
	%B = add i32 4, 2		; <i32> [#uses=2]
	call void @foo2( i32 %B )
	br i1 %c, label %Loop, label %Out
Out:		; preds = %Loop
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}

; CHECK: @test4
; CHECK: call
; CHECK: sdiv
; CHECK: ret
define i32 @test4(i32 %x, i32 %y) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %n.01 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  call void @foo_may_call_exit(i32 0)
  %div = sdiv i32 %x, %y
  %add = add nsw i32 %n.01, %div
  %inc = add nsw i32 %i.02, 1
  %cmp = icmp slt i32 %inc, 10000
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  %n.0.lcssa = phi i32 [ %add, %for.body ]
  ret i32 %n.0.lcssa
}

declare void @foo_may_call_exit(i32)

