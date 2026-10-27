; Check-first flattens the guarded control flow, so a load inside a guarded
; block is speculated on every lane. Bail out unless it is dereferenceable for
; the whole trip count, as %p may only be dereferenceable where flags[i] != 0.
;
;   for (i = 0; i < 1024; i++)
;     if (flags[i]) {
;       A[i] = p[i];
;       if (data[i] == 42)
;         break;
;     }

; RUN: opt -S < %s -p loop-vectorize -force-vector-width=4 -enable-check-first-early-exit-vectorization -pass-remarks-analysis=loop-vectorize -disable-output 2>&1 | FileCheck %s --check-prefix=REMARK
; RUN: opt -S < %s -p loop-vectorize -force-vector-width=4 -enable-check-first-early-exit-vectorization | FileCheck %s

; REMARK: loop not vectorized: Load in a conditionally-executed block is not provably dereferenceable for the whole trip count

; CHECK-NOT: vector.check
; CHECK-NOT: vector.body

@A = global [1024 x i32] zeroinitializer
@flags = global [1024 x i32] zeroinitializer
@data = global [1024 x i32] zeroinitializer

define void @guarded_faulting_load(ptr %p) {
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %latch ]
  %gflags = getelementptr inbounds [1024 x i32], ptr @flags, i64 0, i64 %iv
  %lflags = load i32, ptr %gflags, align 4
  %cflags = icmp ne i32 %lflags, 0
  br i1 %cflags, label %guard, label %latch

guard:
  %gp = getelementptr inbounds i32, ptr %p, i64 %iv
  %lp = load i32, ptr %gp, align 4
  %ga = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %iv
  store i32 %lp, ptr %ga, align 4
  %gdata = getelementptr inbounds [1024 x i32], ptr @data, i64 0, i64 %iv
  %ldata = load i32, ptr %gdata, align 4
  %cdata = icmp eq i32 %ldata, 42
  br i1 %cdata, label %exit, label %latch

latch:
  %iv.next = add nuw nsw i64 %iv, 1
  %done = icmp eq i64 %iv.next, 1024
  br i1 %done, label %ret, label %for.body

exit:
  br label %ret

ret:
  ret void
}
