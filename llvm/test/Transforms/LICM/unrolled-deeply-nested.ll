; Test that LICM correctly detects conflicting accesses to memory in deeply
; nested subloops. This works in the legacy PM due to a special retained map of
; alias information for inner loops, and in the new PM it is recomputed for each
; loop.
;
; RUN: opt -S -aa-pipeline=basic-aa -passes='require<opt-remark-emit>,loop(licm)' < %s | FileCheck %s
; RUN: opt -S -basicaa -licm < %s | FileCheck %s

define i32 @test(i32* %a, i64 %n.0, i64 %n.0.0, i64 %n.0.0.0, i64 %n.0.0.0.0) nounwind uwtable readonly {
; CHECK-LABEL: define i32 @test
entry:
  %b = alloca i32
  %c = alloca i32
  %a.i8 = bitcast i32* %a to i8*
  %b.i8 = bitcast i32* %b to i8*
  %c.i8 = bitcast i32* %c to i8*
  br label %l.0.header
; CHECK: %b = alloca i32
; CHECK: %c = alloca i32
; CHECK: %[[AI8:.*]] = bitcast i32* %a to i8*
; CHECK: %[[BI8:.*]] = bitcast i32* %b to i8*
; CHECK: %[[CI8:.*]] = bitcast i32* %c to i8*
; CHECK-NOT: load
; CHECK: br

l.0.header:
  %iv.0 = phi i64 [ %iv.0.next, %l.0.latch ], [ 0, %entry ]
  %iv.0.next = add i64 %iv.0, 1
  %exitcond.0 = icmp eq i64 %iv.0.next, %n.0
  %a.val = load i32, i32* %a
  store i32 %a.val, i32* %b
  %c.val = trunc i64 %iv.0 to i32
  store i32 %c.val, i32* %c
  br label %l.0.0.header
; CHECK: %[[AV:.*]] = load i32, i32* %a
; CHECK: store i32 %[[AV]], i32* %b
; CHECK: %[[CT:.*]] = trunc i64 {{.*}} to i32
; CHECK: store i32 %[[CT]], i32* %c
; CHECK: br

l.0.0.header:
  %iv.0.0 = phi i64 [ %iv.0.0.next, %l.0.0.latch ], [ 0, %l.0.header ]
  %iv.0.0.next = add i64 %iv.0.0, 1
  %exitcond.0.0 = icmp eq i64 %iv.0.0.next, %n.0.0
  br label %l.0.0.0.header
; CHECK: br

l.0.0.0.header:
  %iv.0.0.0 = phi i64 [ %iv.0.0.0.next, %l.0.0.0.header ], [ 0, %l.0.0.header ]
  %iv.0.0.0.next = add i64 %iv.0.0.0, 1
  %exitcond.0.0.0 = icmp eq i64 %iv.0.0.0.next, %n.0.0.0
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %a.i8, i8* %c.i8, i64 4, i1 false)
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* %b.i8, i8* %c.i8, i64 4, i1 false)
  br i1 %exitcond.0.0.0, label %l.0.0.0.header, label %l.0.0.latch
; CHECK: call void @llvm.memcpy.{{.*}}(i8* %[[AI8]], i8* %[[CI8]], i64 4
; CHECK: call void @llvm.memcpy.{{.*}}(i8* %[[BI8]], i8* %[[CI8]], i64 4
; CHECK: br

l.0.0.latch:
  br i1 %exitcond.0.0, label %l.0.0.header, label %l.0.latch
; CHECK: br

l.0.latch:
  %b.val = load i32, i32* %b
  br i1 %exitcond.0, label %exit, label %l.0.header
; CHECK: %[[BV:.*]] = load i32, i32* %b
; CHECK: br

exit:
  %result.lcssa = phi i32 [ %b.val, %l.0.latch ]
  ret i32 %b.val
; CHECK: %[[LCSSA:.*]] = phi i32 [ %[[BV]], %{{.*}} ]
; CHECK: ret i32 %[[LCSSA]]
}

declare void @llvm.memcpy.p0i8.p0i8.i64(i8* nocapture writeonly, i8* nocapture readonly, i64, i1)
