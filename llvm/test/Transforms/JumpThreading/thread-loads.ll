; RUN: opt < %s -jump-threading -S | FileCheck %s
; RUN: opt < %s -aa-pipeline=basic-aa -passes=jump-threading -S | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:32:64-f32:32:32-f64:32:64-v64:64:64-v128:128:128-a0:0:64-f80:128:128"
target triple = "i386-apple-darwin7"

; Test that we can thread through the block with the partially redundant load (%2).
; rdar://6402033
define i32 @test1(i32* %P) nounwind {
; CHECK-LABEL: @test1(
entry:
	%0 = tail call i32 (...) @f1() nounwind		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1, label %bb

bb:		; preds = %entry
; CHECK: bb1.thread:
; CHECK: store
; CHECK: br label %bb3
	store i32 42, i32* %P, align 4
	br label %bb1

bb1:		; preds = %entry, %bb
	%res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]		; <i32> [#uses=2]
	%2 = load i32, i32* %P, align 4		; <i32> [#uses=1]
	%3 = icmp sgt i32 %2, 36		; <i1> [#uses=1]
	br i1 %3, label %bb3, label %bb2

bb2:		; preds = %bb1
	%4 = tail call i32 (...) @f2() nounwind		; <i32> [#uses=0]
	ret i32 %res.0

bb3:		; preds = %bb1
; CHECK: bb3:
; CHECK: %res.02 = phi i32 [ 1, %bb1.thread ], [ 0, %bb1 ]
; CHECK: ret i32 %res.02
	ret i32 %res.0
}

declare i32 @f1(...)

declare i32 @f2(...)


;; Check that we preserve TBAA information.
; rdar://11039258

define i32 @test2(i32* %P) nounwind {
; CHECK-LABEL: @test2(
entry:
	%0 = tail call i32 (...) @f1() nounwind		; <i32> [#uses=1]
	%1 = icmp eq i32 %0, 0		; <i1> [#uses=1]
	br i1 %1, label %bb1, label %bb

bb:		; preds = %entry
; CHECK: bb1.thread:
; CHECK: store{{.*}}, !tbaa !0
; CHECK: br label %bb3
	store i32 42, i32* %P, align 4, !tbaa !0
	br label %bb1

bb1:		; preds = %entry, %bb
	%res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
	%2 = load i32, i32* %P, align 4, !tbaa !0
	%3 = icmp sgt i32 %2, 36
	br i1 %3, label %bb3, label %bb2

bb2:		; preds = %bb1
	%4 = tail call i32 (...) @f2() nounwind
	ret i32 %res.0

bb3:		; preds = %bb1
; CHECK: bb3:
; CHECK: %res.02 = phi i32 [ 1, %bb1.thread ], [ 0, %bb1 ]
; CHECK: ret i32 %res.02
	ret i32 %res.0
}

define i32 @test3(i8** %x, i1 %f) {
; Correctly thread loads of different (but compatible) types, placing bitcasts
; as necessary in the predecessors. This is especially tricky because the same
; predecessor ends up with two entries in the PHI node and they must share
; a single cast.
; CHECK-LABEL: @test3(
entry:
  %0 = bitcast i8** %x to i32**
  %1 = load i32*, i32** %0, align 8
  br i1 %f, label %if.end57, label %if.then56
; CHECK: %[[LOAD:.*]] = load i32*, i32**
; CHECK: %[[CAST:.*]] = bitcast i32* %[[LOAD]] to i8*

if.then56:
  br label %if.end57

if.end57:
  %2 = load i8*, i8** %x, align 8
  %tobool59 = icmp eq i8* %2, null
  br i1 %tobool59, label %return, label %if.then60
; CHECK: %[[PHI:.*]] = phi i8* [ %[[CAST]], %[[PRED:[^ ]+]] ], [ %[[CAST]], %[[PRED]] ]
; CHECK-NEXT: %[[CMP:.*]] = icmp eq i8* %[[PHI]], null
; CHECK-NEXT: br i1 %[[CMP]]

if.then60:
  ret i32 42

return:
  ret i32 13
}

define i32 @test4(i32* %P) {
; CHECK-LABEL: @test4(
entry:
  %v0 = tail call i32 (...) @f1()
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %bb1, label %bb

bb:
; CHECK: bb1.thread:
; CHECK: store atomic
; CHECK: br label %bb3
  store atomic i32 42, i32* %P unordered, align 4
  br label %bb1

bb1:
; CHECK: bb1:
; CHECK-NOT: phi
; CHECK: load atomic
  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
  %v2 = load atomic i32, i32* %P unordered, align 4
  %v3 = icmp sgt i32 %v2, 36
  br i1 %v3, label %bb3, label %bb2

bb2:
  %v4 = tail call i32 (...) @f2()
  ret i32 %res.0

bb3:
  ret i32 %res.0
}

define i32 @test5(i32* %P) {
; Negative test

; CHECK-LABEL: @test5(
entry:
  %v0 = tail call i32 (...) @f1()
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %bb1, label %bb

bb:
; CHECK: bb:
; CHECK-NEXT:   store atomic i32 42, i32* %P release, align 4
; CHECK-NEXT:   br label %bb1
  store atomic i32 42, i32* %P release, align 4
  br label %bb1

bb1:
; CHECK: bb1:
; CHECK-NEXT:  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
; CHECK-NEXT:  %v2 = load atomic i32, i32* %P acquire, align 4
; CHECK-NEXT:  %v3 = icmp sgt i32 %v2, 36
; CHECK-NEXT:  br i1 %v3, label %bb3, label %bb2

  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
  %v2 = load atomic i32, i32* %P acquire, align 4
  %v3 = icmp sgt i32 %v2, 36
  br i1 %v3, label %bb3, label %bb2

bb2:
  %v4 = tail call i32 (...) @f2()
  ret i32 %res.0

bb3:
  ret i32 %res.0
}

define i32 @test6(i32* %P) {
; Negative test

; CHECK-LABEL: @test6(
entry:
  %v0 = tail call i32 (...) @f1()
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %bb1, label %bb

bb:
; CHECK: bb:
; CHECK-NEXT:   store i32 42, i32* %P
; CHECK-NEXT:   br label %bb1
  store i32 42, i32* %P
  br label %bb1

bb1:
; CHECK: bb1:
; CHECK-NEXT:  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
; CHECK-NEXT:  %v2 = load atomic i32, i32* %P acquire, align 4
; CHECK-NEXT:  %v3 = icmp sgt i32 %v2, 36
; CHECK-NEXT:  br i1 %v3, label %bb3, label %bb2

  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
  %v2 = load atomic i32, i32* %P acquire, align 4
  %v3 = icmp sgt i32 %v2, 36
  br i1 %v3, label %bb3, label %bb2

bb2:
  %v4 = tail call i32 (...) @f2()
  ret i32 %res.0

bb3:
  ret i32 %res.0
}

define i32 @test7(i32* %P) {
; Negative test

; CHECK-LABEL: @test7(
entry:
  %v0 = tail call i32 (...) @f1()
  %v1 = icmp eq i32 %v0, 0
  br i1 %v1, label %bb1, label %bb

bb:
; CHECK: bb:
; CHECK-NEXT:   %val = load i32, i32* %P
; CHECK-NEXT:   br label %bb1
  %val = load i32, i32* %P
  br label %bb1

bb1:
; CHECK: bb1:
; CHECK-NEXT:  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
; CHECK-NEXT:  %v2 = load atomic i32, i32* %P acquire, align 4
; CHECK-NEXT:  %v3 = icmp sgt i32 %v2, 36
; CHECK-NEXT:  br i1 %v3, label %bb3, label %bb2

  %res.0 = phi i32 [ 1, %bb ], [ 0, %entry ]
  %v2 = load atomic i32, i32* %P acquire, align 4
  %v3 = icmp sgt i32 %v2, 36
  br i1 %v3, label %bb3, label %bb2

bb2:
  %v4 = tail call i32 (...) @f2()
  ret i32 %res.0

bb3:
  ret i32 %res.0
}

; Make sure we merge the aliasing metadata. We keep the range metadata for the
; first load, as it dominates the second load. Hence we can eliminate the
; branch.
define void @test8(i32*, i32*, i32*) {
; CHECK-LABEL: @test8(
; CHECK: %a = load i32, i32* %0, align 4, !range ![[RANGE4:[0-9]+]]
; CHECK-NEXT: store i32 %a
; CHECK-NEXT: %xxx = tail call i32 (...) @f1()
; CHECK-NEXT: ret void
  %a = load i32, i32* %0, !tbaa !0, !range !4, !alias.scope !9, !noalias !10
  %b = load i32, i32* %0, !range !5
  store i32 %a, i32* %1
  %c = icmp eq i32 %b, 8
  br i1 %c, label %ret1, label %ret2

ret1:
  ret void

ret2:
  %xxx = tail call i32 (...) @f1() nounwind
  ret void
}

; Make sure we merge/PRE aliasing metadata correctly.  That means that
; we need to remove metadata from the existing load, and add appropriate
; metadata to the newly inserted load.
define void @test9(i32*, i32*, i32*, i1 %c) {
; CHECK-LABEL: @test9(
  br i1 %c, label %d1, label %d2

; CHECK: d1:
; CHECK-NEXT: %a = load i32, i32* %0, align 4{{$}}
d1:
  %a = load i32, i32* %0, !range !4, !alias.scope !9, !noalias !10
  br label %d3

; CHECK: d2:
; CHECK-NEXT: %xxxx = tail call i32 (...) @f1()
; CHECK-NEXT: %b.pr = load i32, i32* %0, align 4, !tbaa !0{{$}}
d2:
  %xxxx = tail call i32 (...) @f1() nounwind
  br label %d3

d3:
  %p = phi i32 [ 1, %d2 ], [ %a, %d1 ]
  %b = load i32, i32* %0, !tbaa !0
  store i32 %p, i32* %1
  %c2 = icmp eq i32 %b, 8
  br i1 %c2, label %ret1, label %ret2

ret1:
  ret void

ret2:
  %xxx = tail call i32 (...) @f1() nounwind
  ret void
}

define i32 @fn_noalias(i1 %c2,i64* noalias %P, i64* noalias %P2) {
; CHECK-LABEL: @fn_noalias
; CHECK-LABEL: cond1:
; CHECK: %[[LD1:.*]] = load i64, i64* %P
; CHECK: br i1 %c, label %[[THREAD:.*]], label %end
; CHECK-LABEL: cond2:
; CHECK: %[[LD2:.*]] = load i64, i64* %P
; CHECK-LABEL: cond3:
; CHECK: %[[PHI:.*]] = phi i64 [ %[[LD1]], %[[THREAD]] ], [ %[[LD2]], %cond2 ]
; CHECK: call void @fn3(i64 %[[PHI]])
entry:
  br i1 %c2, label %cond2, label %cond1

cond1:
  %l1 = load i64, i64* %P
  store i64 42, i64* %P2
  %c = icmp eq i64 %l1, 0
  br i1 %c, label %cond2, label %end

cond2:
  %l2 = load i64, i64* %P
  call void @fn2(i64 %l2)
  %c3 = icmp eq i64 %l2,  0
  br i1 %c3, label %cond3, label %end

cond3:
  call void @fn3(i64 %l2)
  br label %end

end:
  ret i32 0
}

; This tests if we can thread from %sw.bb.i to %do.body.preheader.i67 through
; %sw.bb21.i. To make this happen, %l2 should be detected as a partically
; redundant load with %l3 across the store to %phase in %sw.bb21.i.

%struct.NEXT_MOVE = type { i32, i32, i32* }
@hash_move = unnamed_addr global [65 x i32] zeroinitializer, align 4
@current_move = internal global [65 x i32] zeroinitializer, align 4
@last = internal unnamed_addr global [65 x i32*] zeroinitializer, align 8
@next_status = internal unnamed_addr global [65 x %struct.NEXT_MOVE] zeroinitializer, align 8
define fastcc i32 @Search(i64 %idxprom.i, i64 %idxprom.i89, i32 %c) {
; CHECK-LABEL: @Search
; CHECK-LABEL: sw.bb.i:
; CHECK: %[[LD1:.*]] = load i32, i32* %arrayidx185, align 4
; CHECK: %[[C1:.*]] = icmp eq i32 %[[LD1]], 0
; CHECK: br i1 %[[C1]], label %sw.bb21.i.thread, label %if.then.i64
; CHECK-LABEL: sw.bb21.i.thread:
; CHECK: br label %[[THREAD_TO:.*]]
; CHECK-LABEL: sw.bb21.i:
; CHECK: %[[LD2:.*]] = load i32, i32* %arrayidx185, align 4
; CHECK: %[[C2:.*]] = icmp eq i32 %[[LD2]], 0
; CHECK:br i1 %[[C2]], label %[[THREAD_TO]], label %cleanup
entry:
  %arrayidx185 = getelementptr inbounds [65 x i32], [65 x i32]* @hash_move, i64 0, i64 %idxprom.i
  %arrayidx307 = getelementptr inbounds [65 x i32], [65 x i32]* @current_move, i64 0, i64 %idxprom.i
  %arrayidx89 = getelementptr inbounds [65 x i32*], [65 x i32*]* @last, i64 0, i64 %idxprom.i
  %phase = getelementptr inbounds [65 x %struct.NEXT_MOVE], [65 x %struct.NEXT_MOVE]* @next_status, i64 0, i64 %idxprom.i, i32 0
  br label %cond.true282

cond.true282:
  switch i32 %c, label %sw.default.i [
    i32 1, label %sw.bb.i
    i32 0, label %sw.bb21.i
  ]

sw.default.i:
  br label %cleanup

sw.bb.i:
  %call.i62 = call fastcc i32* @GenerateCheckEvasions()
  store i32* %call.i62, i32** %arrayidx89, align 8
  %l2 = load i32, i32* %arrayidx185, align 4
  %tobool.i63 = icmp eq i32 %l2, 0
  br i1 %tobool.i63, label %sw.bb21.i, label %if.then.i64

if.then.i64:                                      ; preds = %sw.bb.i
  store i32 7, i32* %phase, align 8
  store i32 %l2, i32* %arrayidx307, align 4
  %call16.i = call fastcc i32 @ValidMove(i32 %l2)
  %tobool17.i = icmp eq i32 %call16.i, 0
  br i1 %tobool17.i, label %if.else.i65, label %cleanup

if.else.i65:
  call void @f65()
  br label %sw.bb21.i

sw.bb21.i:
  store i32 10, i32* %phase, align 8
  %l3= load i32, i32* %arrayidx185, align 4
  %tobool27.i = icmp eq i32 %l3, 0
  br i1 %tobool27.i, label %do.body.preheader.i67, label %cleanup

do.body.preheader.i67:
  call void @f67()
  ret  i32 67

cleanup:
  call void @Cleanup()
  ret  i32 0
}

declare fastcc i32* @GenerateCheckEvasions()
declare fastcc i32 @ValidMove(i32 %move)
declare void @f67()
declare void @Cleanup()
declare void @f65()

define i32 @fn_SinglePred(i1 %c2,i64* %P) {
; CHECK-LABEL: @fn_SinglePred
; CHECK-LABEL: entry:
; CHECK: %[[L1:.*]] = load i64, i64* %P
; CHECK: br i1 %c, label %cond3, label %cond1
; CHECK-LABEL: cond2:
; CHECK-NOT: load
; CHECK: %[[PHI:.*]] = phi i64 [ %[[L1]], %cond1 ]
; CHECK: call void @fn2(i64 %[[PHI]])
; CHECK: br label %end
; CHECK-LABEL: cond3:
; CHECK: call void @fn2(i64 %l1)
; CHECK: call void @fn3(i64 %l1)

entry:
  %l1 = load i64, i64* %P
  %c = icmp eq i64 %l1, 0
  br i1 %c, label %cond2, label %cond1

cond1:
  br i1 %c2, label %cond2, label %end

cond2:
  %l2 = load i64, i64* %P
  call void @fn2(i64 %l2)
  %c3 = icmp eq i64 %l2,  0
  br i1 %c3, label %cond3, label %end

cond3:
  call void @fn3(i64 %l2)
  br label %end

end:
  ret i32 0
}

define i32 @fn_SinglePredMultihop(i1 %c1, i1 %c2,i64* %P) {
; CHECK-LABEL: @fn_SinglePredMultihop
; CHECK-LABEL: entry:
; CHECK: %[[L1:.*]] = load i64, i64* %P
; CHECK: br i1 %c0, label %cond3, label %cond0
; CHECK-LABEL: cond2:
; CHECK-NOT: load
; CHECK: %[[PHI:.*]] = phi i64 [ %[[L1]], %cond1 ]
; CHECK: call void @fn2(i64 %[[PHI]])
; CHECK: br label %end
; CHECK-LABEL: cond3:
; CHECK: call void @fn2(i64 %l1)
; CHECK: call void @fn3(i64 %l1)

entry:
  %l1 = load i64, i64* %P
  %c0 = icmp eq i64 %l1, 0
  br i1 %c0, label %cond2, label %cond0

cond0:
  br i1 %c1, label %cond1, label %end

cond1:
  br i1 %c2, label %cond2, label %end

cond2:
  %l2 = load i64, i64* %P
  call void @fn2(i64 %l2)
  %c3 = icmp eq i64 %l2,  0
  br i1 %c3, label %cond3, label %end

cond3:
  call void @fn3(i64 %l2)
  br label %end

end:
  ret i32 0
}

declare void @fn2(i64)
declare void @fn3(i64)


; Make sure we phi-translate and make the partially redundant load in
; merge fully redudant and then we can jump-thread the block with the
; store.
;
; CHECK-LABEL: define i32 @phi_translate_partial_redundant_loads(i32 %0, i32* %1, i32* %2
; CHECK: merge.thread:
; CHECK: store
; CHECK: br label %left_x
;
; CHECK: left_x:
; CHECK-NEXT: ret i32 20
define i32 @phi_translate_partial_redundant_loads(i32, i32*, i32*)  {
  %cmp0 = icmp ne i32 %0, 0
  br i1 %cmp0, label %left, label %right

left:
  store i32 1, i32* %1, align 4
  br label %merge

right:
  br label %merge

merge:
  %phiptr = phi i32* [ %1, %left ], [ %2, %right ]
  %newload = load i32, i32* %phiptr, align 4
  %cmp1 = icmp slt i32 %newload, 5
  br i1 %cmp1, label %left_x, label %right_x

left_x:
  ret i32 20

right_x:
  ret i32 10
}

; CHECK: ![[RANGE4]] = !{i32 0, i32 1}

!0 = !{!3, !3, i64 0}
!1 = !{!"omnipotent char", !2}
!2 = !{!"Simple C/C++ TBAA"}
!3 = !{!"int", !1}
!4 = !{ i32 0, i32 1 }
!5 = !{ i32 8, i32 10 }
!6 = !{!6}
!7 = !{!7, !6}
!8 = !{!8, !6}
!9 = !{!7}
!10 = !{!8}
