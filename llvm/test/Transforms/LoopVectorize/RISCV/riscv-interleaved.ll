; REQUIRES: asserts
; RUN: opt -passes=loop-vectorize -mtriple riscv64-linux-gnu \
; RUN:   -mattr=+v -debug-only=loop-vectorize --disable-output \
; RUN:   -scalable-vectorization=off -S < %s 2>&1 | FileCheck %s

; CHECK-LABEL: foo
; CHECK: LV: IC is 2
; CHECK: %{{.*}} = add {{.*}}, 16
; CHECK: %{{.*}} = add <8 x i32> %{{.*}}, splat (i32 8)

define void @foo(i32 signext %n, ptr nocapture %A) {
entry:
  %cmp5 = icmp sgt i32 %n, 0
  br i1 %cmp5, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:
  %wide.trip.count = zext i32 %n to i64
  br label %for.body

for.cond.cleanup.loopexit:
  br label %for.cond.cleanup

for.cond.cleanup:
  ret void

for.body:
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i64 %indvars.iv
  %0 = trunc i64 %indvars.iv to i32
  store i32 %0, ptr %arrayidx, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup.loopexit, label %for.body
}
