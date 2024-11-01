; RUN: opt < %s -force-vector-interleave=1 -store-to-load-forwarding-conflict-detection=false -passes=loop-vectorize,dce,instcombine -S | FileCheck %s

target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"
target triple = "aarch64--linux-gnu"

%struct.pair = type { i32, i32 }

; Check vectorization of interleaved access groups with positive dependence
; distances. In this test, the maximum safe dependence distance for
; vectorization is 16 bytes. Normally, this would lead to a maximum VF of 4.
; However, for interleaved groups, the effective VF is VF * IF, where IF is the
; interleave factor. Here, the maximum safe dependence distance is recomputed
; as 16 / IF bytes, resulting in VF=2. Since IF=2, we should generate <4 x i32>
; loads and stores instead of <8 x i32> accesses.
;
; Note: LAA's conflict detection optimization has to be disabled for this test
;       to be vectorized.

; struct pair {
;   int x;
;   int y;
; };
;
; void max_vf(struct pair *restrict p) {
;   for (int i = 0; i < 1000; i++) {
;     p[i + 2].x = p[i].x
;     p[i + 2].y = p[i].y
;   }
; }

; CHECK-LABEL: @max_vf
; CHECK: load <4 x i32>
; CHECK: store <4 x i32>

define void @max_vf(%struct.pair* noalias nocapture %p) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 0, %entry ], [ %i.next, %for.body ]
  %0 = add nuw nsw i64 %i, 2
  %p_i.x = getelementptr inbounds %struct.pair, %struct.pair* %p, i64 %i, i32 0
  %p_i_plus_2.x = getelementptr inbounds %struct.pair, %struct.pair* %p, i64 %0, i32 0
  %1 = load i32, i32* %p_i.x, align 4
  store i32 %1, i32* %p_i_plus_2.x, align 4
  %p_i.y = getelementptr inbounds %struct.pair, %struct.pair* %p, i64 %i, i32 1
  %p_i_plus_2.y = getelementptr inbounds %struct.pair, %struct.pair* %p, i64 %0, i32 1
  %2 = load i32, i32* %p_i.y, align 4
  store i32 %2, i32* %p_i_plus_2.y, align 4
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp eq i64 %i.next, 1000
  br i1 %cond, label %for.exit, label %for.body

for.exit:
  ret void
}
