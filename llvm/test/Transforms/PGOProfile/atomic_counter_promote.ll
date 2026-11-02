; RUN: opt < %s -passes=instrprof -instrprof-atomic-counter-update-all -do-counter-promotion -S | FileCheck %s

; CHECK: define i32 @foo(i32 %n) {
; CHECK: entry:
; CHECK:   atomicrmw add {{.*}}ptr @__profc_foo
;
; CHECK: for.cond.for.cond.cleanup_crit_edge:
; CHECK-NOT: br
; CHECK:      atomicrmw add {{.*}}ptr @__profc_foo
; CHECK-NEXT: atomicrmw add {{.*}}ptr @__profc_foo

@__profn_foo = private constant [3 x i8] c"foo"

define i32 @foo(i32 %n) {
entry:
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 1124680652598534200, i32 3, i32 2)
  %cmp16 = icmp slt i32 0, %n
  br i1 %cmp16, label %for.cond1.preheader.lr.ph, label %for.cond.cleanup

for.cond1.preheader.lr.ph:
  br label %for.cond1.preheader

for.cond1.preheader:
  %i.018 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %inc6, %for.cond.cleanup3 ]
  %x.017 = phi i32 [ 0, %for.cond1.preheader.lr.ph ], [ %x.1.lcssa, %for.cond.cleanup3 ]
  %cmp213 = icmp slt i32 0, %n
  br i1 %cmp213, label %for.body4.lr.ph, label %for.cond.cleanup3

for.body4.lr.ph:
  br label %for.body4

for.cond.for.cond.cleanup_crit_edge:
  %split19 = phi i32 [ %x.1.lcssa, %for.cond.cleanup3 ]
  br label %for.cond.cleanup

for.cond.cleanup:
  %x.0.lcssa = phi i32 [ %split19, %for.cond.for.cond.cleanup_crit_edge ], [ 0, %entry ]
  ret i32 %x.0.lcssa

for.cond1.for.cond.cleanup3_crit_edge:
  %split = phi i32 [ %add, %for.body4 ]
  br label %for.cond.cleanup3

for.cond.cleanup3:
  %x.1.lcssa = phi i32 [ %split, %for.cond1.for.cond.cleanup3_crit_edge ], [ %x.017, %for.cond1.preheader ]
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 1124680652598534200, i32 3, i32 1)
  %inc6 = add nuw nsw i32 %i.018, 1
  %cmp = icmp slt i32 %inc6, %n
  br i1 %cmp, label %for.cond1.preheader, label %for.cond.for.cond.cleanup_crit_edge

for.body4:
  %j.015 = phi i32 [ 0, %for.body4.lr.ph ], [ %inc, %for.body4 ]
  %x.114 = phi i32 [ %x.017, %for.body4.lr.ph ], [ %add, %for.body4 ]
  call void @llvm.instrprof.increment(ptr @__profn_foo, i64 1124680652598534200, i32 3, i32 0)
  %add = add nsw i32 %x.114, %j.015
  %inc = add nuw nsw i32 %j.015, 1
  %cmp2 = icmp slt i32 %inc, %n
  br i1 %cmp2, label %for.body4, label %for.cond1.for.cond.cleanup3_crit_edge
}

declare void @llvm.instrprof.increment(ptr, i64, i32, i32)
