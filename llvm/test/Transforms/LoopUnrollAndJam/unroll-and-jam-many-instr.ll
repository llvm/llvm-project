; RUN: opt -passes=loop-unroll-and-jam  -allow-unroll-and-jam -unroll-and-jam-count=4 < %s -S | FileCheck %s
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"

@a = common dso_local local_unnamed_addr global i32 0, align 4
@b = common dso_local local_unnamed_addr global i8 0, align 1
@e = common dso_local local_unnamed_addr global i64 0, align 8
@c = common dso_local local_unnamed_addr global ptr null, align 8
@g = common dso_local local_unnamed_addr global i64 0, align 8
@f = common dso_local local_unnamed_addr global i32 0, align 4

; Check that the loop won't be UAJ and it will finish.
; This loops triggers a exponential explosion of the algorithm.
; CHECK-LABEL: test1
; CHECK: entry
; CHECK: for.cond1.preheader.lr.ph
; CHECK: for.cond1.preheader
; CHECK: for.cond13.preheader
; CHECK: for.cond4.preheader
; CHECK: for.cond.for.end27_crit_edge
; CHECK: for.end27
define dso_local void @test1(i32 %i) {
entry:
  %0 = load i32, ptr @a, align 4, !tbaa !1
  %tobool40 = icmp eq i32 %0, 0
  br i1 %tobool40, label %for.end27, label %for.cond1.preheader.lr.ph

for.cond1.preheader.lr.ph:                        ; preds = %entry
  %1 = load ptr, ptr @c, align 8, !tbaa !5
  br label %for.cond1.preheader

for.cond1.preheader:                              ; preds = %for.cond1.preheader.lr.ph, %for.cond13.preheader
  %i.addr.041 = phi i32 [ %i, %for.cond1.preheader.lr.ph ], [ %xor.8.9, %for.cond13.preheader ]
  br label %for.cond4.preheader

for.cond13.preheader:                             ; preds = %for.cond4.preheader
  %tobool21 = icmp ne i32 %i.addr.041, 0
  %lor.ext = zext i1 %tobool21 to i32
  %xor = xor i32 %i.addr.041, %lor.ext
  %tobool21.1 = icmp ne i32 %xor, 0
  %lor.ext.1 = zext i1 %tobool21.1 to i32
  %xor.1 = xor i32 %xor, %lor.ext.1
  %tobool21.2 = icmp ne i32 %xor.1, 0
  %lor.ext.2 = zext i1 %tobool21.2 to i32
  %xor.2 = xor i32 %xor.1, %lor.ext.2
  %tobool21.3 = icmp ne i32 %xor.2, 0
  %lor.ext.3 = zext i1 %tobool21.3 to i32
  %xor.3 = xor i32 %xor.2, %lor.ext.3
  %tobool21.4 = icmp ne i32 %xor.3, 0
  %lor.ext.4 = zext i1 %tobool21.4 to i32
  %xor.4 = xor i32 %xor.3, %lor.ext.4
  %tobool21.5 = icmp ne i32 %xor.4, 0
  %lor.ext.5 = zext i1 %tobool21.5 to i32
  %xor.5 = xor i32 %xor.4, %lor.ext.5
  %tobool21.6 = icmp ne i32 %xor.5, 0
  %lor.ext.6 = zext i1 %tobool21.6 to i32
  %xor.6 = xor i32 %xor.5, %lor.ext.6
  %tobool21.7 = icmp ne i32 %xor.6, 0
  %lor.ext.7 = zext i1 %tobool21.7 to i32
  %xor.7 = xor i32 %xor.6, %lor.ext.7
  %tobool21.8 = icmp ne i32 %xor.7, 0
  %lor.ext.8 = zext i1 %tobool21.8 to i32
  %xor.8 = xor i32 %xor.7, %lor.ext.8
  %tobool21.142 = icmp ne i32 %xor.8, 0
  %lor.ext.143 = zext i1 %tobool21.142 to i32
  %xor.144 = xor i32 %xor.8, %lor.ext.143
  %tobool21.1.1 = icmp ne i32 %xor.144, 0
  %lor.ext.1.1 = zext i1 %tobool21.1.1 to i32
  %xor.1.1 = xor i32 %xor.144, %lor.ext.1.1
  %tobool21.2.1 = icmp ne i32 %xor.1.1, 0
  %lor.ext.2.1 = zext i1 %tobool21.2.1 to i32
  %xor.2.1 = xor i32 %xor.1.1, %lor.ext.2.1
  %tobool21.3.1 = icmp ne i32 %xor.2.1, 0
  %lor.ext.3.1 = zext i1 %tobool21.3.1 to i32
  %xor.3.1 = xor i32 %xor.2.1, %lor.ext.3.1
  %tobool21.4.1 = icmp ne i32 %xor.3.1, 0
  %lor.ext.4.1 = zext i1 %tobool21.4.1 to i32
  %xor.4.1 = xor i32 %xor.3.1, %lor.ext.4.1
  %tobool21.5.1 = icmp ne i32 %xor.4.1, 0
  %lor.ext.5.1 = zext i1 %tobool21.5.1 to i32
  %xor.5.1 = xor i32 %xor.4.1, %lor.ext.5.1
  %tobool21.6.1 = icmp ne i32 %xor.5.1, 0
  %lor.ext.6.1 = zext i1 %tobool21.6.1 to i32
  %xor.6.1 = xor i32 %xor.5.1, %lor.ext.6.1
  %tobool21.7.1 = icmp ne i32 %xor.6.1, 0
  %lor.ext.7.1 = zext i1 %tobool21.7.1 to i32
  %xor.7.1 = xor i32 %xor.6.1, %lor.ext.7.1
  %tobool21.8.1 = icmp ne i32 %xor.7.1, 0
  %lor.ext.8.1 = zext i1 %tobool21.8.1 to i32
  %xor.8.1 = xor i32 %xor.7.1, %lor.ext.8.1
  %tobool21.245 = icmp ne i32 %xor.8.1, 0
  %lor.ext.246 = zext i1 %tobool21.245 to i32
  %xor.247 = xor i32 %xor.8.1, %lor.ext.246
  %tobool21.1.2 = icmp ne i32 %xor.247, 0
  %lor.ext.1.2 = zext i1 %tobool21.1.2 to i32
  %xor.1.2 = xor i32 %xor.247, %lor.ext.1.2
  %tobool21.2.2 = icmp ne i32 %xor.1.2, 0
  %lor.ext.2.2 = zext i1 %tobool21.2.2 to i32
  %xor.2.2 = xor i32 %xor.1.2, %lor.ext.2.2
  %tobool21.3.2 = icmp ne i32 %xor.2.2, 0
  %lor.ext.3.2 = zext i1 %tobool21.3.2 to i32
  %xor.3.2 = xor i32 %xor.2.2, %lor.ext.3.2
  %tobool21.4.2 = icmp ne i32 %xor.3.2, 0
  %lor.ext.4.2 = zext i1 %tobool21.4.2 to i32
  %xor.4.2 = xor i32 %xor.3.2, %lor.ext.4.2
  %tobool21.5.2 = icmp ne i32 %xor.4.2, 0
  %lor.ext.5.2 = zext i1 %tobool21.5.2 to i32
  %xor.5.2 = xor i32 %xor.4.2, %lor.ext.5.2
  %tobool21.6.2 = icmp ne i32 %xor.5.2, 0
  %lor.ext.6.2 = zext i1 %tobool21.6.2 to i32
  %xor.6.2 = xor i32 %xor.5.2, %lor.ext.6.2
  %tobool21.7.2 = icmp ne i32 %xor.6.2, 0
  %lor.ext.7.2 = zext i1 %tobool21.7.2 to i32
  %xor.7.2 = xor i32 %xor.6.2, %lor.ext.7.2
  %tobool21.8.2 = icmp ne i32 %xor.7.2, 0
  %lor.ext.8.2 = zext i1 %tobool21.8.2 to i32
  %xor.8.2 = xor i32 %xor.7.2, %lor.ext.8.2
  %tobool21.348 = icmp ne i32 %xor.8.2, 0
  %lor.ext.349 = zext i1 %tobool21.348 to i32
  %xor.350 = xor i32 %xor.8.2, %lor.ext.349
  %tobool21.1.3 = icmp ne i32 %xor.350, 0
  %lor.ext.1.3 = zext i1 %tobool21.1.3 to i32
  %xor.1.3 = xor i32 %xor.350, %lor.ext.1.3
  %tobool21.2.3 = icmp ne i32 %xor.1.3, 0
  %lor.ext.2.3 = zext i1 %tobool21.2.3 to i32
  %xor.2.3 = xor i32 %xor.1.3, %lor.ext.2.3
  %tobool21.3.3 = icmp ne i32 %xor.2.3, 0
  %lor.ext.3.3 = zext i1 %tobool21.3.3 to i32
  %xor.3.3 = xor i32 %xor.2.3, %lor.ext.3.3
  %tobool21.4.3 = icmp ne i32 %xor.3.3, 0
  %lor.ext.4.3 = zext i1 %tobool21.4.3 to i32
  %xor.4.3 = xor i32 %xor.3.3, %lor.ext.4.3
  %tobool21.5.3 = icmp ne i32 %xor.4.3, 0
  %lor.ext.5.3 = zext i1 %tobool21.5.3 to i32
  %xor.5.3 = xor i32 %xor.4.3, %lor.ext.5.3
  %tobool21.6.3 = icmp ne i32 %xor.5.3, 0
  %lor.ext.6.3 = zext i1 %tobool21.6.3 to i32
  %xor.6.3 = xor i32 %xor.5.3, %lor.ext.6.3
  %tobool21.7.3 = icmp ne i32 %xor.6.3, 0
  %lor.ext.7.3 = zext i1 %tobool21.7.3 to i32
  %xor.7.3 = xor i32 %xor.6.3, %lor.ext.7.3
  %tobool21.8.3 = icmp ne i32 %xor.7.3, 0
  %lor.ext.8.3 = zext i1 %tobool21.8.3 to i32
  %xor.8.3 = xor i32 %xor.7.3, %lor.ext.8.3
  %tobool21.451 = icmp ne i32 %xor.8.3, 0
  %lor.ext.452 = zext i1 %tobool21.451 to i32
  %xor.453 = xor i32 %xor.8.3, %lor.ext.452
  %tobool21.1.4 = icmp ne i32 %xor.453, 0
  %lor.ext.1.4 = zext i1 %tobool21.1.4 to i32
  %xor.1.4 = xor i32 %xor.453, %lor.ext.1.4
  %tobool21.2.4 = icmp ne i32 %xor.1.4, 0
  %lor.ext.2.4 = zext i1 %tobool21.2.4 to i32
  %xor.2.4 = xor i32 %xor.1.4, %lor.ext.2.4
  %tobool21.3.4 = icmp ne i32 %xor.2.4, 0
  %lor.ext.3.4 = zext i1 %tobool21.3.4 to i32
  %xor.3.4 = xor i32 %xor.2.4, %lor.ext.3.4
  %tobool21.4.4 = icmp ne i32 %xor.3.4, 0
  %lor.ext.4.4 = zext i1 %tobool21.4.4 to i32
  %xor.4.4 = xor i32 %xor.3.4, %lor.ext.4.4
  %tobool21.5.4 = icmp ne i32 %xor.4.4, 0
  %lor.ext.5.4 = zext i1 %tobool21.5.4 to i32
  %xor.5.4 = xor i32 %xor.4.4, %lor.ext.5.4
  %tobool21.6.4 = icmp ne i32 %xor.5.4, 0
  %lor.ext.6.4 = zext i1 %tobool21.6.4 to i32
  %xor.6.4 = xor i32 %xor.5.4, %lor.ext.6.4
  %tobool21.7.4 = icmp ne i32 %xor.6.4, 0
  %lor.ext.7.4 = zext i1 %tobool21.7.4 to i32
  %xor.7.4 = xor i32 %xor.6.4, %lor.ext.7.4
  %tobool21.8.4 = icmp ne i32 %xor.7.4, 0
  %lor.ext.8.4 = zext i1 %tobool21.8.4 to i32
  %xor.8.4 = xor i32 %xor.7.4, %lor.ext.8.4
  %tobool21.554 = icmp ne i32 %xor.8.4, 0
  %lor.ext.555 = zext i1 %tobool21.554 to i32
  %xor.556 = xor i32 %xor.8.4, %lor.ext.555
  %tobool21.1.5 = icmp ne i32 %xor.556, 0
  %lor.ext.1.5 = zext i1 %tobool21.1.5 to i32
  %xor.1.5 = xor i32 %xor.556, %lor.ext.1.5
  %tobool21.2.5 = icmp ne i32 %xor.1.5, 0
  %lor.ext.2.5 = zext i1 %tobool21.2.5 to i32
  %xor.2.5 = xor i32 %xor.1.5, %lor.ext.2.5
  %tobool21.3.5 = icmp ne i32 %xor.2.5, 0
  %lor.ext.3.5 = zext i1 %tobool21.3.5 to i32
  %xor.3.5 = xor i32 %xor.2.5, %lor.ext.3.5
  %tobool21.4.5 = icmp ne i32 %xor.3.5, 0
  %lor.ext.4.5 = zext i1 %tobool21.4.5 to i32
  %xor.4.5 = xor i32 %xor.3.5, %lor.ext.4.5
  %tobool21.5.5 = icmp ne i32 %xor.4.5, 0
  %lor.ext.5.5 = zext i1 %tobool21.5.5 to i32
  %xor.5.5 = xor i32 %xor.4.5, %lor.ext.5.5
  %tobool21.6.5 = icmp ne i32 %xor.5.5, 0
  %lor.ext.6.5 = zext i1 %tobool21.6.5 to i32
  %xor.6.5 = xor i32 %xor.5.5, %lor.ext.6.5
  %tobool21.7.5 = icmp ne i32 %xor.6.5, 0
  %lor.ext.7.5 = zext i1 %tobool21.7.5 to i32
  %xor.7.5 = xor i32 %xor.6.5, %lor.ext.7.5
  %tobool21.8.5 = icmp ne i32 %xor.7.5, 0
  %lor.ext.8.5 = zext i1 %tobool21.8.5 to i32
  %xor.8.5 = xor i32 %xor.7.5, %lor.ext.8.5
  %tobool21.657 = icmp ne i32 %xor.8.5, 0
  %lor.ext.658 = zext i1 %tobool21.657 to i32
  %xor.659 = xor i32 %xor.8.5, %lor.ext.658
  %tobool21.1.6 = icmp ne i32 %xor.659, 0
  %lor.ext.1.6 = zext i1 %tobool21.1.6 to i32
  %xor.1.6 = xor i32 %xor.659, %lor.ext.1.6
  %tobool21.2.6 = icmp ne i32 %xor.1.6, 0
  %lor.ext.2.6 = zext i1 %tobool21.2.6 to i32
  %xor.2.6 = xor i32 %xor.1.6, %lor.ext.2.6
  %tobool21.3.6 = icmp ne i32 %xor.2.6, 0
  %lor.ext.3.6 = zext i1 %tobool21.3.6 to i32
  %xor.3.6 = xor i32 %xor.2.6, %lor.ext.3.6
  %tobool21.4.6 = icmp ne i32 %xor.3.6, 0
  %lor.ext.4.6 = zext i1 %tobool21.4.6 to i32
  %xor.4.6 = xor i32 %xor.3.6, %lor.ext.4.6
  %tobool21.5.6 = icmp ne i32 %xor.4.6, 0
  %lor.ext.5.6 = zext i1 %tobool21.5.6 to i32
  %xor.5.6 = xor i32 %xor.4.6, %lor.ext.5.6
  %tobool21.6.6 = icmp ne i32 %xor.5.6, 0
  %lor.ext.6.6 = zext i1 %tobool21.6.6 to i32
  %xor.6.6 = xor i32 %xor.5.6, %lor.ext.6.6
  %tobool21.7.6 = icmp ne i32 %xor.6.6, 0
  %lor.ext.7.6 = zext i1 %tobool21.7.6 to i32
  %xor.7.6 = xor i32 %xor.6.6, %lor.ext.7.6
  %tobool21.8.6 = icmp ne i32 %xor.7.6, 0
  %lor.ext.8.6 = zext i1 %tobool21.8.6 to i32
  %xor.8.6 = xor i32 %xor.7.6, %lor.ext.8.6
  %tobool21.760 = icmp ne i32 %xor.8.6, 0
  %lor.ext.761 = zext i1 %tobool21.760 to i32
  %xor.762 = xor i32 %xor.8.6, %lor.ext.761
  %tobool21.1.7 = icmp ne i32 %xor.762, 0
  %lor.ext.1.7 = zext i1 %tobool21.1.7 to i32
  %xor.1.7 = xor i32 %xor.762, %lor.ext.1.7
  %tobool21.2.7 = icmp ne i32 %xor.1.7, 0
  %lor.ext.2.7 = zext i1 %tobool21.2.7 to i32
  %xor.2.7 = xor i32 %xor.1.7, %lor.ext.2.7
  %tobool21.3.7 = icmp ne i32 %xor.2.7, 0
  %lor.ext.3.7 = zext i1 %tobool21.3.7 to i32
  %xor.3.7 = xor i32 %xor.2.7, %lor.ext.3.7
  %tobool21.4.7 = icmp ne i32 %xor.3.7, 0
  %lor.ext.4.7 = zext i1 %tobool21.4.7 to i32
  %xor.4.7 = xor i32 %xor.3.7, %lor.ext.4.7
  %tobool21.5.7 = icmp ne i32 %xor.4.7, 0
  %lor.ext.5.7 = zext i1 %tobool21.5.7 to i32
  %xor.5.7 = xor i32 %xor.4.7, %lor.ext.5.7
  %tobool21.6.7 = icmp ne i32 %xor.5.7, 0
  %lor.ext.6.7 = zext i1 %tobool21.6.7 to i32
  %xor.6.7 = xor i32 %xor.5.7, %lor.ext.6.7
  %tobool21.7.7 = icmp ne i32 %xor.6.7, 0
  %lor.ext.7.7 = zext i1 %tobool21.7.7 to i32
  %xor.7.7 = xor i32 %xor.6.7, %lor.ext.7.7
  %tobool21.8.7 = icmp ne i32 %xor.7.7, 0
  %lor.ext.8.7 = zext i1 %tobool21.8.7 to i32
  %xor.8.7 = xor i32 %xor.7.7, %lor.ext.8.7
  %tobool21.863 = icmp ne i32 %xor.8.7, 0
  %lor.ext.864 = zext i1 %tobool21.863 to i32
  %xor.865 = xor i32 %xor.8.7, %lor.ext.864
  %tobool21.1.8 = icmp ne i32 %xor.865, 0
  %lor.ext.1.8 = zext i1 %tobool21.1.8 to i32
  %xor.1.8 = xor i32 %xor.865, %lor.ext.1.8
  %tobool21.2.8 = icmp ne i32 %xor.1.8, 0
  %lor.ext.2.8 = zext i1 %tobool21.2.8 to i32
  %xor.2.8 = xor i32 %xor.1.8, %lor.ext.2.8
  %tobool21.3.8 = icmp ne i32 %xor.2.8, 0
  %lor.ext.3.8 = zext i1 %tobool21.3.8 to i32
  %xor.3.8 = xor i32 %xor.2.8, %lor.ext.3.8
  %tobool21.4.8 = icmp ne i32 %xor.3.8, 0
  %lor.ext.4.8 = zext i1 %tobool21.4.8 to i32
  %xor.4.8 = xor i32 %xor.3.8, %lor.ext.4.8
  %tobool21.5.8 = icmp ne i32 %xor.4.8, 0
  %lor.ext.5.8 = zext i1 %tobool21.5.8 to i32
  %xor.5.8 = xor i32 %xor.4.8, %lor.ext.5.8
  %tobool21.6.8 = icmp ne i32 %xor.5.8, 0
  %lor.ext.6.8 = zext i1 %tobool21.6.8 to i32
  %xor.6.8 = xor i32 %xor.5.8, %lor.ext.6.8
  %tobool21.7.8 = icmp ne i32 %xor.6.8, 0
  %lor.ext.7.8 = zext i1 %tobool21.7.8 to i32
  %xor.7.8 = xor i32 %xor.6.8, %lor.ext.7.8
  %tobool21.8.8 = icmp ne i32 %xor.7.8, 0
  %lor.ext.8.8 = zext i1 %tobool21.8.8 to i32
  %xor.8.8 = xor i32 %xor.7.8, %lor.ext.8.8
  %tobool21.9 = icmp ne i32 %xor.8.8, 0
  %lor.ext.9 = zext i1 %tobool21.9 to i32
  %xor.9 = xor i32 %xor.8.8, %lor.ext.9
  %tobool21.1.9 = icmp ne i32 %xor.9, 0
  %lor.ext.1.9 = zext i1 %tobool21.1.9 to i32
  %xor.1.9 = xor i32 %xor.9, %lor.ext.1.9
  %tobool21.2.9 = icmp ne i32 %xor.1.9, 0
  %lor.ext.2.9 = zext i1 %tobool21.2.9 to i32
  %xor.2.9 = xor i32 %xor.1.9, %lor.ext.2.9
  %tobool21.3.9 = icmp ne i32 %xor.2.9, 0
  %lor.ext.3.9 = zext i1 %tobool21.3.9 to i32
  %xor.3.9 = xor i32 %xor.2.9, %lor.ext.3.9
  %tobool21.4.9 = icmp ne i32 %xor.3.9, 0
  %lor.ext.4.9 = zext i1 %tobool21.4.9 to i32
  %xor.4.9 = xor i32 %xor.3.9, %lor.ext.4.9
  %tobool21.5.9 = icmp ne i32 %xor.4.9, 0
  %lor.ext.5.9 = zext i1 %tobool21.5.9 to i32
  %xor.5.9 = xor i32 %xor.4.9, %lor.ext.5.9
  %tobool21.6.9 = icmp ne i32 %xor.5.9, 0
  %lor.ext.6.9 = zext i1 %tobool21.6.9 to i32
  %xor.6.9 = xor i32 %xor.5.9, %lor.ext.6.9
  %tobool21.7.9 = icmp ne i32 %xor.6.9, 0
  %lor.ext.7.9 = zext i1 %tobool21.7.9 to i32
  %xor.7.9 = xor i32 %xor.6.9, %lor.ext.7.9
  %tobool21.8.9 = icmp ne i32 %xor.7.9, 0
  %lor.ext.8.9 = zext i1 %tobool21.8.9 to i32
  %xor.8.9 = xor i32 %xor.7.9, %lor.ext.8.9
  store i32 10, ptr @f, align 4, !tbaa !1
  %2 = load i32, ptr @a, align 4, !tbaa !1
  %tobool = icmp eq i32 %2, 0
  br i1 %tobool, label %for.cond.for.end27_crit_edge, label %for.cond1.preheader

for.cond4.preheader:                              ; preds = %for.cond1.preheader, %for.cond4.preheader
  %j.035 = phi i32 [ 9, %for.cond1.preheader ], [ %dec11, %for.cond4.preheader ]
  %3 = load i8, ptr @b, align 1, !tbaa !7
  %conv = zext i8 %3 to i32
  %cmp = icmp sgt i32 %i.addr.041, %conv
  %conv7 = zext i1 %cmp to i32
  store i32 %conv7, ptr %1, align 4, !tbaa !1
  %4 = load i8, ptr @b, align 1, !tbaa !7
  %conv.1 = zext i8 %4 to i32
  %cmp.1 = icmp sgt i32 %i.addr.041, %conv.1
  %conv7.1 = zext i1 %cmp.1 to i32
  store i32 %conv7.1, ptr %1, align 4, !tbaa !1
  %5 = load i8, ptr @b, align 1, !tbaa !7
  %conv.2 = zext i8 %5 to i32
  %cmp.2 = icmp sgt i32 %i.addr.041, %conv.2
  %conv7.2 = zext i1 %cmp.2 to i32
  store i32 %conv7.2, ptr %1, align 4, !tbaa !1
  %6 = load i8, ptr @b, align 1, !tbaa !7
  %conv.3 = zext i8 %6 to i32
  %cmp.3 = icmp sgt i32 %i.addr.041, %conv.3
  %conv7.3 = zext i1 %cmp.3 to i32
  store i32 %conv7.3, ptr %1, align 4, !tbaa !1
  %7 = load i8, ptr @b, align 1, !tbaa !7
  %conv.4 = zext i8 %7 to i32
  %cmp.4 = icmp sgt i32 %i.addr.041, %conv.4
  %conv7.4 = zext i1 %cmp.4 to i32
  store i32 %conv7.4, ptr %1, align 4, !tbaa !1
  %8 = load i8, ptr @b, align 1, !tbaa !7
  %conv.5 = zext i8 %8 to i32
  %cmp.5 = icmp sgt i32 %i.addr.041, %conv.5
  %conv7.5 = zext i1 %cmp.5 to i32
  store i32 %conv7.5, ptr %1, align 4, !tbaa !1
  %9 = load i8, ptr @b, align 1, !tbaa !7
  %conv.6 = zext i8 %9 to i32
  %cmp.6 = icmp sgt i32 %i.addr.041, %conv.6
  %conv7.6 = zext i1 %cmp.6 to i32
  store i32 %conv7.6, ptr %1, align 4, !tbaa !1
  %10 = load i8, ptr @b, align 1, !tbaa !7
  %conv.7 = zext i8 %10 to i32
  %cmp.7 = icmp sgt i32 %i.addr.041, %conv.7
  %conv7.7 = zext i1 %cmp.7 to i32
  store i32 %conv7.7, ptr %1, align 4, !tbaa !1
  %11 = load i8, ptr @b, align 1, !tbaa !7
  %conv.8 = zext i8 %11 to i32
  %cmp.8 = icmp sgt i32 %i.addr.041, %conv.8
  %conv7.8 = zext i1 %cmp.8 to i32
  store i32 %conv7.8, ptr %1, align 4, !tbaa !1
  %dec11 = add nsw i32 %j.035, -1
  %tobool2 = icmp eq i32 %dec11, 0
  br i1 %tobool2, label %for.cond13.preheader, label %for.cond4.preheader

for.cond.for.end27_crit_edge:                     ; preds = %for.cond13.preheader
  %conv8.le.le = zext i1 %cmp.8 to i64
  store i64 %conv8.le.le, ptr @e, align 8, !tbaa !8
  store i64 10, ptr @g, align 8, !tbaa !8
  br label %for.end27

for.end27:                                        ; preds = %for.cond.for.end27_crit_edge, %entry
  ret void
}

!1 = !{!2, !2, i64 0}
!2 = !{!"int", !3, i64 0}
!3 = !{!"omnipotent char", !4, i64 0}
!4 = !{!"Simple C/C++ TBAA"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !3, i64 0}
!7 = !{!3, !3, i64 0}
!8 = !{!9, !9, i64 0}
!9 = !{!"long", !3, i64 0}
