; RUN: opt < %s -passes='require<loops>,gvn' -enable-load-in-loop-pre=false -S | FileCheck %s
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64"

;void test1(int N, ptr G) {
;  int j;
;  for (j = 0; j < N - 1; j++)
;    G[j] = G[j] + G[j+1];
;}

define void @test1(i32 %N, ptr nocapture %G) nounwind ssp {
; CHECK-LABEL: @test1(
entry:
  %0 = add i32 %N, -1
  %1 = icmp sgt i32 %0, 0
  br i1 %1, label %bb.nph, label %return

bb.nph:
  %tmp = zext i32 %0 to i64
  br label %bb

; CHECK: bb.nph:
; CHECK-NOT: load double, ptr
; CHECK: br label %bb

bb:
  %indvar = phi i64 [ 0, %bb.nph ], [ %tmp6, %bb ]
  %tmp6 = add i64 %indvar, 1
  %scevgep = getelementptr double, ptr %G, i64 %tmp6
  %scevgep7 = getelementptr double, ptr %G, i64 %indvar
  %2 = load double, ptr %scevgep7, align 8
  %3 = load double, ptr %scevgep, align 8
  %4 = fadd double %2, %3
  store double %4, ptr %scevgep7, align 8
  %exitcond = icmp eq i64 %tmp6, %tmp
  br i1 %exitcond, label %return, label %bb

; Both loads should remain in the loop.
; CHECK: bb:
; CHECK: load double, ptr
; CHECK: load double, ptr
; CHECK: br i1 %exitcond

return:
  ret void
}
