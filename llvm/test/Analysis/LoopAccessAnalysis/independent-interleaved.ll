; RUN: opt -passes='print<access-info>' -store-to-load-forwarding-conflict-detection=false  -disable-output  < %s 2>&1 | FileCheck %s

; This test checks that we prove the strided accesses to be independent before
; concluding that there is a forward dependence.

; struct pair {
;   int x;
;   int y;
; };
;
; int independent_interleaved(struct pair *p, int z, int n) {
;   int s = 0;
;   for (int i = 0; i < n; i++) {
;     p[i].y = z;
;     s += p[i].x;
;   }
;   return s;
; }

; CHECK:     for.body:
; CHECK-NOT:     Forward:
; CHECK-NOT:         store i32 %z, ptr %p_i.y, align 8 ->
; CHECK-NOT:         %0 = load i32, ptr %p_i.x, align 8

%pair = type { i32, i32 }
define i32 @independent_interleaved(ptr %p, i64 %n, i32 %z) {
entry:
  br label %for.body

for.body:
  %i = phi i64 [ %i.next, %for.body ], [ 0, %entry ]
  %s = phi i32 [ %1, %for.body ], [ 0, %entry ]
  %p_i.x = getelementptr inbounds %pair, ptr %p, i64 %i, i32 0
  %p_i.y = getelementptr inbounds %pair, ptr %p, i64 %i, i32 1
  store i32 %z, ptr %p_i.y, align 8
  %0 = load i32, ptr %p_i.x, align 8
  %1 = add nsw i32 %0, %s
  %i.next = add nuw nsw i64 %i, 1
  %cond = icmp slt i64 %i.next, %n
  br i1 %cond, label %for.body, label %for.end

for.end:
  %2 = phi i32 [ %1, %for.body ]
  ret i32 %2
}
