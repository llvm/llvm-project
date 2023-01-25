; RUN: opt -passes=loop-simplify -S < %s | FileCheck %s

; CHECK-LABEL: @test1
define void @test1(i32 %n) {
entry:
  br label %while.cond

while.cond:                                       ; preds = %if.then, %if.else, %entry
  %count.0 = phi i32 [ 0, %entry ], [ %add, %if.then ], [ %add2, %if.else ]
  %cmp = icmp ugt i32 %count.0, %n
  br i1 %cmp, label %while.end, label %while.body

while.body:                                       ; preds = %while.cond
  %rem = and i32 %count.0, 1
  %cmp1 = icmp eq i32 %rem, 0
  br i1 %cmp1, label %if.then, label %if.else

if.then:                                          ; preds = %while.body
  %add = add i32 %count.0, 1
  br label %while.cond, !llvm.loop !0

if.else:                                          ; preds = %while.body
  %add2 = add i32 %count.0, 2
  br label %while.cond, !llvm.loop !0

while.end:                                        ; preds = %while.cond
  ret void
}

; CHECK: if.then
; CHECK-NOT: br {{.*}}!llvm.loop{{.*}}

; CHECK: while.cond.backedge:
; CHECK: br label %while.cond, !llvm.loop !0

; CHECK: if.else
; CHECK-NOT: br {{.*}}!llvm.loop{{.*}}

; CHECK-LABEL: @test2
; CHECK: for.body:
; CHECK: br i1 %{{.*}}, label %for.body, label %cleanup.loopexit, !llvm.loop !0
define void @test2(i32 %k)  {
entry:
  %cmp9 = icmp sgt i32 %k, 0
  br i1 %cmp9, label %for.body.preheader, label %cleanup

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.cond:                                         ; preds = %for.body
  %cmp = icmp slt i32 %inc, %k
  br i1 %cmp, label %for.body, label %cleanup.loopexit, !llvm.loop !0

for.body:                                         ; preds = %for.body.preheader, %for.cond
  %i.010 = phi i32 [ %inc, %for.cond ], [ 0, %for.body.preheader ]
  %cmp3 = icmp sgt i32 %i.010, 3
  %inc = add nsw i32 %i.010, 1
  br i1 %cmp3, label %cleanup.loopexit, label %for.cond

cleanup.loopexit:                                 ; preds = %for.body, %for.cond
  br label %cleanup

cleanup:                                          ; preds = %cleanup.loopexit, %entry
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.distribute.enable", i1 true}
; CHECK: !0 = distinct !{!0, !1}
; CHECK: !1 = !{!"llvm.loop.distribute.enable", i1 true}
