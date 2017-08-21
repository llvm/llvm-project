; RUN: opt -loop-vectorize -force-vector-width=4 -S < %s | FileCheck %s

; This is the test case from PR26314.
; When we were retrying dependence checking with memchecks only,
; the loop-invariant access in the inner loop was incorrectly determined to be wrapping
; because it was not strided in the inner loop.
; Improved wrapping detection allows vectorization in the following case.

; #define Z 32
; typedef struct s {
;       int v1[Z];
;       int v2[Z];
;       int v3[Z][Z];
; } s;
;
; void slow_function (s* const obj, int z) {
;    for (int j=0; j<Z; j++) {
;        for (int k=0; k<z; k++) {
;            int x = obj->v1[k] + obj->v2[j];
;            obj->v3[j][k] += x;
;        }
;    }
; }

; CHECK-LABEL: Test
; CHECK: <4 x i64>
; CHECK: <4 x i32>, <4 x i32>
; CHECK: !{!"llvm.loop.isvectorized", i32 1}

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

%struct.s = type { [32 x i32], [32 x i32], [32 x [32 x i32]] }

define void @Test(%struct.s* nocapture %obj, i64 %z) #0 {
  br label %.outer.preheader


.outer.preheader:
  %i = phi i64 [ 0, %0 ], [ %i.next, %.outer ]
  %1 = getelementptr inbounds %struct.s, %struct.s* %obj, i64 0, i32 1, i64 %i
  br label %.inner

.exit:
  ret void
 
.outer:
  %i.next = add nuw nsw i64 %i, 1
  %exitcond.outer = icmp eq i64 %i.next, 32
  br i1 %exitcond.outer, label %.exit, label %.outer.preheader

.inner:
  %j = phi i64 [ 0, %.outer.preheader ], [ %j.next, %.inner ]
  %2 = getelementptr inbounds %struct.s, %struct.s* %obj, i64 0, i32 0, i64 %j
  %3 = load i32, i32* %2
  %4 = load i32, i32* %1
  %5 = add nsw i32 %4, %3
  %6 = getelementptr inbounds %struct.s, %struct.s* %obj, i64 0, i32 2, i64 %i, i64 %j
  %7 = load i32, i32* %6
  %8 = add nsw i32 %5, %7
  store i32 %8, i32* %6  
  %j.next = add nuw nsw i64 %j, 1
  %exitcond.inner = icmp eq i64 %j.next, %z
  br i1 %exitcond.inner, label %.outer, label %.inner
}
