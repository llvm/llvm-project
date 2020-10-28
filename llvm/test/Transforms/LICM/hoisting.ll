; RUN: opt < %s -licm -S | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes='require<opt-remark-emit>,loop(licm)' -S | FileCheck %s
; RUN: opt < %s -licm -enable-mssa-loop-dependency=true -verify-memoryssa -S | FileCheck %s

@X = global i32 0		; <i32*> [#uses=1]

declare void @foo()

declare i32 @llvm.bitreverse.i32(i32)

; This testcase tests for a problem where LICM hoists 
; potentially trapping instructions when they are not guaranteed to execute.
define i32 @test1(i1 %c) {
; CHECK-LABEL: @test1(
	%A = load i32, i32* @X		; <i32> [#uses=2]
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
; CHECK-LABEL: @test2(
; CHECK-NEXT: load i32, i32* @X
; CHECK-NEXT: %B = sdiv i32 4, %A
  %A = load i32, i32* @X
  br label %Loop

Loop:
  ;; Should have hoisted this div!
  %B = sdiv i32 4, %A
  br label %loop2

loop2:
  call void @foo2( i32 %B )
  br i1 %c, label %Loop, label %Out

Out:
  %C = sub i32 %A, %B
  ret i32 %C
}


; This loop invariant instruction should be constant folded, not hoisted.
define i32 @test3(i1 %c) {
; CHECK-LABEL: define i32 @test3(
; CHECK: call void @foo2(i32 6)
	%A = load i32, i32* @X		; <i32> [#uses=2]
	br label %Loop
Loop:
	%B = add i32 4, 2		; <i32> [#uses=2]
	call void @foo2( i32 %B )
	br i1 %c, label %Loop, label %Out
Out:		; preds = %Loop
	%C = sub i32 %A, %B		; <i32> [#uses=1]
	ret i32 %C
}

; CHECK-LABEL: @test4(
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

; PR14854
; CHECK-LABEL: @test5(
; CHECK: extractvalue
; CHECK: br label %tailrecurse
; CHECK: tailrecurse:
; CHECK: ifend:
; CHECK: insertvalue
define { i32*, i32 } @test5(i32 %i, { i32*, i32 } %e) {
entry:
  br label %tailrecurse

tailrecurse:                                      ; preds = %then, %entry
  %i.tr = phi i32 [ %i, %entry ], [ %cmp2, %then ]
  %out = extractvalue { i32*, i32 } %e, 1
  %d = insertvalue { i32*, i32 } %e, i32* null, 0
  %cmp1 = icmp sgt i32 %out, %i.tr
  br i1 %cmp1, label %then, label %ifend

then:                                             ; preds = %tailrecurse
  call void @foo()
  %cmp2 = add i32 %i.tr, 1
  br label %tailrecurse

ifend:                                            ; preds = %tailrecurse
  ret { i32*, i32 } %d
}

; CHECK: define void @test6(float %f)
; CHECK: fneg
; CHECK: br label %for.body
define void @test6(float %f) #2 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  call void @foo_may_call_exit(i32 0)
  %neg = fneg float %f
  call void @use(float %neg)
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, 10000
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
  ret void
}

declare void @use(float)

; CHECK: define i32 @hoist_bitreverse(i32 %0)
; CHECK: bitreverse
; CHECK: br label %header
define i32 @hoist_bitreverse(i32 %0)  {
  br label %header

header:
  %sum = phi i32 [ 0, %1 ], [ %5, %latch ]
  %2 = phi i32 [ 0, %1 ], [ %6, %latch ]
  %3 = icmp slt i32 %2, 1024
  br i1 %3, label %body, label %return

body:
  %4 = call i32 @llvm.bitreverse.i32(i32 %0)
  %5 = add i32 %sum, %4
  br label %latch

latch:
  %6 = add nsw i32 %2, 1
  br label %header

return:
  ret i32 %sum
}

; Can neither sink nor hoist
define i32 @test_volatile(i1 %c) {
; CHECK-LABEL: @test_volatile(
; CHECK-LABEL: Loop:
; CHECK: load volatile i32, i32* @X
; CHECK-LABEL: Out:
  br label %Loop

Loop:
  %A = load volatile i32, i32* @X
  br i1 %c, label %Loop, label %Out

Out:
  ret i32 %A
}


declare {}* @llvm.invariant.start.p0i8(i64, i8* nocapture) nounwind readonly
declare void @llvm.invariant.end.p0i8({}*, i64, i8* nocapture) nounwind
declare void @escaping.invariant.start({}*) nounwind
; invariant.start dominates the load, and in this scope, the
; load is invariant. So, we can hoist the `addrld` load out of the loop.
define i32 @test_fence(i8* %addr, i32 %n, i8* %volatile) {
; CHECK-LABEL: @test_fence
; CHECK-LABEL: entry
; CHECK: invariant.start
; CHECK: %addrld = load atomic i32, i32* %addr.i unordered, align 8
; CHECK: br label %loop
entry: 
  %gep = getelementptr inbounds i8, i8* %addr, i64 8
  %addr.i = bitcast i8* %gep to i32 *
  store atomic i32 5, i32 * %addr.i unordered, align 8
  fence release
  %invst = call {}* @llvm.invariant.start.p0i8(i64 4, i8* %gep)
  br label %loop

loop:
  %indvar = phi i32 [ %indvar.next, %loop ], [ 0, %entry ]
  %sum = phi i32 [ %sum.next, %loop ], [ 0, %entry ]
  %volload = load atomic i8, i8* %volatile unordered, align 8
  fence acquire
  %volchk = icmp eq i8 %volload, 0
  %addrld = load atomic i32, i32* %addr.i unordered, align 8
  %sel = select i1 %volchk, i32 0, i32 %addrld
  %sum.next = add i32 %sel, %sum
  %indvar.next = add i32 %indvar, 1
  %cond = icmp slt i32 %indvar.next, %n
  br i1 %cond, label %loop, label %loopexit

loopexit:
  ret i32 %sum
}



; Same as test above, but the load is no longer invariant (presence of
; invariant.end). We cannot hoist the addrld out of loop.
define i32 @test_fence1(i8* %addr, i32 %n, i8* %volatile) {
; CHECK-LABEL: @test_fence1
; CHECK-LABEL: entry
; CHECK: invariant.start
; CHECK-NEXT: invariant.end
; CHECK-NEXT: br label %loop
entry:
  %gep = getelementptr inbounds i8, i8* %addr, i64 8
  %addr.i = bitcast i8* %gep to i32 *
  store atomic i32 5, i32 * %addr.i unordered, align 8
  fence release
  %invst = call {}* @llvm.invariant.start.p0i8(i64 4, i8* %gep)
  call void @llvm.invariant.end.p0i8({}* %invst, i64 4, i8* %gep)
  br label %loop

loop:
  %indvar = phi i32 [ %indvar.next, %loop ], [ 0, %entry ]
  %sum = phi i32 [ %sum.next, %loop ], [ 0, %entry ]
  %volload = load atomic i8, i8* %volatile unordered, align 8
  fence acquire
  %volchk = icmp eq i8 %volload, 0
  %addrld = load atomic i32, i32* %addr.i unordered, align 8
  %sel = select i1 %volchk, i32 0, i32 %addrld
  %sum.next = add i32 %sel, %sum
  %indvar.next = add i32 %indvar, 1
  %cond = icmp slt i32 %indvar.next, %n
  br i1 %cond, label %loop, label %loopexit

loopexit:
  ret i32 %sum
}

; same as test above, but instead of invariant.end, we have the result of
; invariant.start escaping through a call. We cannot hoist the load.
define i32 @test_fence2(i8* %addr, i32 %n, i8* %volatile) {
; CHECK-LABEL: @test_fence2
; CHECK-LABEL: entry
; CHECK-NOT: load
; CHECK: br label %loop
entry:
  %gep = getelementptr inbounds i8, i8* %addr, i64 8
  %addr.i = bitcast i8* %gep to i32 *
  store atomic i32 5, i32 * %addr.i unordered, align 8
  fence release
  %invst = call {}* @llvm.invariant.start.p0i8(i64 4, i8* %gep)
  call void @escaping.invariant.start({}* %invst)
  br label %loop

loop:
  %indvar = phi i32 [ %indvar.next, %loop ], [ 0, %entry ]
  %sum = phi i32 [ %sum.next, %loop ], [ 0, %entry ]
  %volload = load atomic i8, i8* %volatile unordered, align 8
  fence acquire
  %volchk = icmp eq i8 %volload, 0
  %addrld = load atomic i32, i32* %addr.i unordered, align 8
  %sel = select i1 %volchk, i32 0, i32 %addrld
  %sum.next = add i32 %sel, %sum
  %indvar.next = add i32 %indvar, 1
  %cond = icmp slt i32 %indvar.next, %n
  br i1 %cond, label %loop, label %loopexit

loopexit:
  ret i32 %sum
}

; FIXME: invariant.start dominates the load, and in this scope, the
; load is invariant. So, we can hoist the `addrld` load out of the loop.
; Consider the loadoperand addr.i bitcasted before being passed to
; invariant.start
define i32 @test_fence3(i32* %addr, i32 %n, i8* %volatile) {
; CHECK-LABEL: @test_fence3
; CHECK-LABEL: entry
; CHECK: invariant.start
; CHECK-NOT: %addrld = load atomic i32, i32* %addr.i unordered, align 8
; CHECK: br label %loop
entry: 
  %addr.i = getelementptr inbounds i32, i32* %addr, i64 8
  %gep = bitcast i32* %addr.i to i8 *
  store atomic i32 5, i32 * %addr.i unordered, align 8
  fence release
  %invst = call {}* @llvm.invariant.start.p0i8(i64 4, i8* %gep)
  br label %loop

loop:
  %indvar = phi i32 [ %indvar.next, %loop ], [ 0, %entry ]
  %sum = phi i32 [ %sum.next, %loop ], [ 0, %entry ]
  %volload = load atomic i8, i8* %volatile unordered, align 8
  fence acquire
  %volchk = icmp eq i8 %volload, 0
  %addrld = load atomic i32, i32* %addr.i unordered, align 8
  %sel = select i1 %volchk, i32 0, i32 %addrld
  %sum.next = add i32 %sel, %sum
  %indvar.next = add i32 %indvar, 1
  %cond = icmp slt i32 %indvar.next, %n
  br i1 %cond, label %loop, label %loopexit

loopexit:
  ret i32 %sum
}

; We should not hoist the addrld out of the loop.
define i32 @test_fence4(i32* %addr, i32 %n, i8* %volatile) {
; CHECK-LABEL: @test_fence4
; CHECK-LABEL: entry
; CHECK-NOT: %addrld = load atomic i32, i32* %addr.i unordered, align 8
; CHECK: br label %loop
entry: 
  %addr.i = getelementptr inbounds i32, i32* %addr, i64 8
  %gep = bitcast i32* %addr.i to i8 *
  br label %loop

loop:
  %indvar = phi i32 [ %indvar.next, %loop ], [ 0, %entry ]
  %sum = phi i32 [ %sum.next, %loop ], [ 0, %entry ]
  store atomic i32 5, i32 * %addr.i unordered, align 8
  fence release
  %invst = call {}* @llvm.invariant.start.p0i8(i64 4, i8* %gep)
  %volload = load atomic i8, i8* %volatile unordered, align 8
  fence acquire
  %volchk = icmp eq i8 %volload, 0
  %addrld = load atomic i32, i32* %addr.i unordered, align 8
  %sel = select i1 %volchk, i32 0, i32 %addrld
  %sum.next = add i32 %sel, %sum
  %indvar.next = add i32 %indvar, 1
  %cond = icmp slt i32 %indvar.next, %n
  br i1 %cond, label %loop, label %loopexit

loopexit:
  ret i32 %sum
}

; We can't hoist the invariant load out of the loop because
; the marker is given a variable size (-1).
define i32 @test_fence5(i8* %addr, i32 %n, i8* %volatile) {
; CHECK-LABEL: @test_fence5
; CHECK-LABEL: entry
; CHECK: invariant.start
; CHECK-NOT: %addrld = load atomic i32, i32* %addr.i unordered, align 8
; CHECK: br label %loop
entry:
  %gep = getelementptr inbounds i8, i8* %addr, i64 8
  %addr.i = bitcast i8* %gep to i32 *
  store atomic i32 5, i32 * %addr.i unordered, align 8
  fence release
  %invst = call {}* @llvm.invariant.start.p0i8(i64 -1, i8* %gep)
  br label %loop

loop:
  %indvar = phi i32 [ %indvar.next, %loop ], [ 0, %entry ]
  %sum = phi i32 [ %sum.next, %loop ], [ 0, %entry ]
  %volload = load atomic i8, i8* %volatile unordered, align 8
  fence acquire
  %volchk = icmp eq i8 %volload, 0
  %addrld = load atomic i32, i32* %addr.i unordered, align 8
  %sel = select i1 %volchk, i32 0, i32 %addrld
  %sum.next = add i32 %sel, %sum
  %indvar.next = add i32 %indvar, 1
  %cond = icmp slt i32 %indvar.next, %n
  br i1 %cond, label %loop, label %loopexit

loopexit:
  ret i32 %sum
}
