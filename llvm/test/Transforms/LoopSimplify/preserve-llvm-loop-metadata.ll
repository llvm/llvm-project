; RUN: opt -loop-simplify -S < %s | FileCheck %s

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


!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.distribute.enable", i1 true}
; CHECK: !0 = distinct !{!0, !1}
; CHECK: !1 = !{!"llvm.loop.distribute.enable", i1 true}
