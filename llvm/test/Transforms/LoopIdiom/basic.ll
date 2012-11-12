; RUN: opt -basicaa -loop-idiom < %s -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"
target triple = "x86_64-apple-darwin10.0.0"

define void @test1(i8* %Base, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test1
; CHECK: call void @llvm.memset.p0i8.i64(i8* %Base, i8 0, i64 %Size, i32 1, i1 false)
; CHECK-NOT: store
}

; This is a loop that was rotated but where the blocks weren't merged.  This
; shouldn't perturb us.
define void @test1a(i8* %Base, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body.cont ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  br label %for.body.cont
for.body.cont:
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test1a
; CHECK: call void @llvm.memset.p0i8.i64(i8* %Base, i8 0, i64 %Size, i32 1, i1 false)
; CHECK-NOT: store
}


define void @test2(i32* %Base, i64 %Size) nounwind ssp {
entry:
  %cmp10 = icmp eq i64 %Size, 0
  br i1 %cmp10, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %add.ptr.i = getelementptr i32* %Base, i64 %i.011
  store i32 16843009, i32* %add.ptr.i, align 4
  %inc = add nsw i64 %i.011, 1
  %exitcond = icmp eq i64 %inc, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test2
; CHECK: br i1 %cmp10,
; CHECK: %0 = mul i64 %Size, 4
; CHECK: call void @llvm.memset.p0i8.i64(i8* %Base1, i8 1, i64 %0, i32 4, i1 false)
; CHECK-NOT: store
}

; This is a case where there is an extra may-aliased store in the loop, we can't
; promote the memset.
define void @test3(i32* %Base, i64 %Size, i8 *%MayAlias) nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i.011 = phi i64 [ %inc, %for.body ], [ 0, %entry ]
  %add.ptr.i = getelementptr i32* %Base, i64 %i.011
  store i32 16843009, i32* %add.ptr.i, align 4
  
  store i8 42, i8* %MayAlias
  %inc = add nsw i64 %i.011, 1
  %exitcond = icmp eq i64 %inc, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %entry
  ret void
; CHECK: @test3
; CHECK-NOT: memset
; CHECK: ret void
}


;; TODO: We should be able to promote this memset.  Not yet though.
define void @test4(i8* %Base) nounwind ssp {
bb.nph:                                           ; preds = %entry
  %Base100 = getelementptr i8* %Base, i64 1000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  
  ;; Store beyond the range memset, should be safe to promote.
  store i8 42, i8* %Base100
  
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK-TODO: @test4
; CHECK-TODO: call void @llvm.memset.p0i8.i64(i8* %Base, i8 0, i64 100, i32 1, i1 false)
; CHECK-TODO-NOT: store
}

; This can't be promoted: the memset is a store of a loop variant value.
define void @test5(i8* %Base, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  
  %V = trunc i64 %indvar to i8
  store i8 %V, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test5
; CHECK-NOT: memset
; CHECK: ret void
}


;; memcpy formation
define void @test6(i64 %Size) nounwind ssp {
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  %DestI = getelementptr i8* %Dest, i64 %indvar
  %V = load i8* %I.0.014, align 1
  store i8 %V, i8* %DestI, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test6
; CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* %Dest, i8* %Base, i64 %Size, i32 1, i1 false)
; CHECK-NOT: store
; CHECK: ret void
}


; This is a loop that was rotated but where the blocks weren't merged.  This
; shouldn't perturb us.
define void @test7(i8* %Base, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body.cont ]
  br label %for.body.cont
for.body.cont:
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  store i8 0, i8* %I.0.014, align 1
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test7
; CHECK: call void @llvm.memset.p0i8.i64(i8* %Base, i8 0, i64 %Size, i32 1, i1 false)
; CHECK-NOT: store
}

; This is a loop should not be transformed, it only executes one iteration.
define void @test8(i64* %Ptr, i64 %Size) nounwind ssp {
bb.nph:                                           ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %PI = getelementptr i64* %Ptr, i64 %indvar
  store i64 0, i64 *%PI
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 1
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test8
; CHECK: store i64 0, i64* %PI
}

declare i8* @external(i8*)

;; This cannot be transformed into a memcpy, because the read-from location is
;; mutated by the loop.
define void @test9(i64 %Size) nounwind ssp {
bb.nph:
  %Base = alloca i8, i32 10000
  %Dest = alloca i8, i32 10000
  
  %BaseAlias = call i8* @external(i8* %Base)
  br label %for.body

for.body:                                         ; preds = %bb.nph, %for.body
  %indvar = phi i64 [ 0, %bb.nph ], [ %indvar.next, %for.body ]
  %I.0.014 = getelementptr i8* %Base, i64 %indvar
  %DestI = getelementptr i8* %Dest, i64 %indvar
  %V = load i8* %I.0.014, align 1
  store i8 %V, i8* %DestI, align 1

  ;; This store can clobber the input.
  store i8 4, i8* %BaseAlias
 
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, %Size
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  ret void
; CHECK: @test9
; CHECK-NOT: llvm.memcpy
; CHECK: ret void
}

; Two dimensional nested loop should be promoted to one big memset.
define void @test10(i8* %X) nounwind ssp {
entry:
  br label %bb.nph

bb.nph:                                           ; preds = %entry, %for.inc10
  %i.04 = phi i32 [ 0, %entry ], [ %inc12, %for.inc10 ]
  br label %for.body5

for.body5:                                        ; preds = %for.body5, %bb.nph
  %j.02 = phi i32 [ 0, %bb.nph ], [ %inc, %for.body5 ]
  %mul = mul nsw i32 %i.04, 100
  %add = add nsw i32 %j.02, %mul
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i8* %X, i64 %idxprom
  store i8 0, i8* %arrayidx, align 1
  %inc = add nsw i32 %j.02, 1
  %cmp4 = icmp eq i32 %inc, 100
  br i1 %cmp4, label %for.inc10, label %for.body5

for.inc10:                                        ; preds = %for.body5
  %inc12 = add nsw i32 %i.04, 1
  %cmp = icmp eq i32 %inc12, 100
  br i1 %cmp, label %for.end13, label %bb.nph

for.end13:                                        ; preds = %for.inc10
  ret void
; CHECK: @test10
; CHECK: entry:
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %X, i8 0, i64 10000, i32 1, i1 false)
; CHECK-NOT: store
; CHECK: ret void
}

; On darwin10 (which is the triple in this .ll file) this loop can be turned
; into a memset_pattern call.
; rdar://9009151
define void @test11_pattern(i32* nocapture %P) nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i32* %P, i64 %indvar
  store i32 1, i32* %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
; CHECK: @test11_pattern
; CHECK-NEXT: entry:
; CHECK-NEXT: bitcast
; CHECK-NEXT: memset_pattern
; CHECK-NOT: store
; CHECK: ret void
}

; Store of null should turn into memset of zero.
define void @test12(i32** nocapture %P) nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i32** %P, i64 %indvar
  store i32* null, i32** %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
; CHECK: @test12
; CHECK-NEXT: entry:
; CHECK-NEXT: bitcast
; CHECK-NEXT: call void @llvm.memset.p0i8.i64(i8* %P1, i8 0, i64 80000, i32 4, i1 false)
; CHECK-NOT: store
; CHECK: ret void
}

@G = global i32 5

; This store-of-address loop can be turned into a memset_pattern call.
; rdar://9009151
define void @test13_pattern(i32** nocapture %P) nounwind ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvar = phi i64 [ 0, %entry ], [ %indvar.next, %for.body ]
  %arrayidx = getelementptr i32** %P, i64 %indvar
  store i32* @G, i32** %arrayidx, align 4
  %indvar.next = add i64 %indvar, 1
  %exitcond = icmp eq i64 %indvar.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
; CHECK: @test13_pattern
; CHECK-NEXT: entry:
; CHECK-NEXT: bitcast
; CHECK-NEXT: memset_pattern
; CHECK-NOT: store
; CHECK: ret void
}



; PR9815 - This is a partial overlap case that cannot be safely transformed
; into a memcpy.
@g_50 = global [7 x i32] [i32 0, i32 0, i32 0, i32 0, i32 1, i32 0, i32 0], align 16

define i32 @test14() nounwind {
entry:
  br label %for.body

for.body:                                         ; preds = %for.inc, %for.body.lr.ph
  %tmp5 = phi i32 [ %inc, %for.body ], [ 0, %entry ]
  %add = add nsw i32 %tmp5, 4
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds [7 x i32]* @g_50, i32 0, i64 %idxprom
  %tmp2 = load i32* %arrayidx, align 4
  %add4 = add nsw i32 %tmp5, 5
  %idxprom5 = sext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds [7 x i32]* @g_50, i32 0, i64 %idxprom5
  store i32 %tmp2, i32* %arrayidx6, align 4
  %inc = add nsw i32 %tmp5, 1
  %cmp = icmp slt i32 %inc, 2
  br i1 %cmp, label %for.body, label %for.end

for.end:                                          ; preds = %for.inc
  %tmp8 = load i32* getelementptr inbounds ([7 x i32]* @g_50, i32 0, i64 6), align 4
  ret i32 %tmp8
; CHECK: @test14
; CHECK: for.body:
; CHECK: load i32
; CHECK: store i32
; CHECK: br i1 %cmp

}

define void @PR14241(i32* %s, i64 %size) {
; Ensure that we don't form a memcpy for strided loops. Briefly, when we taught
; LoopIdiom about memmove and strided loops, this got miscompiled into a memcpy
; instead of a memmove. If we get the memmove transform back, this will catch
; regressions.
;
; CHECK: @PR14241

entry:
  %end.idx = add i64 %size, -1
  %end.ptr = getelementptr inbounds i32* %s, i64 %end.idx
  br label %while.body
; CHECK-NOT: memcpy
;
; FIXME: When we regain the ability to form a memmove here, this test should be
; reversed and turned into a positive assertion.
; CHECK-NOT: memmove

while.body:
  %phi.ptr = phi i32* [ %s, %entry ], [ %next.ptr, %while.body ]
  %src.ptr = getelementptr inbounds i32* %phi.ptr, i64 1
  %val = load i32* %src.ptr, align 4
; CHECK: load
  %dst.ptr = getelementptr inbounds i32* %phi.ptr, i64 0
  store i32 %val, i32* %dst.ptr, align 4
; CHECK: store
  %next.ptr = getelementptr inbounds i32* %phi.ptr, i64 1
  %cmp = icmp eq i32* %next.ptr, %end.ptr
  br i1 %cmp, label %exit, label %while.body

exit:
  ret void
; CHECK: ret void
}
