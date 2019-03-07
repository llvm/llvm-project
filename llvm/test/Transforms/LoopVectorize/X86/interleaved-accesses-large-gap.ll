; RUN: opt < %s  -loop-vectorize -mtriple x86_64 -S | FileCheck %s

%struct.ST4 = type { i32, i32, i32, i32 }

; The gaps between the memory access in this function are too large for
; interleaving.

; Test from https://bugs.chromium.org/p/oss-fuzz/issues/detail?id=7560
define void @test1(%struct.ST4* noalias %B) {
; CHECK-LABEL: @test1
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label %for.body

; CHECK-LABEL: for.body:
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
; CHECK: store i32
; CHECK-NOT: store
;
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %p1 = getelementptr inbounds %struct.ST4, %struct.ST4* %B, i64 %indvars.iv, i32 0
  store i32 65536, i32* %p1, align 4
  %p2 = getelementptr i32, i32* %p1, i32 -2147483648
  store i32 65536, i32* %p2, align 4
  %p3 = getelementptr inbounds %struct.ST4, %struct.ST4* %B, i64 %indvars.iv, i32 2
  store i32 10, i32* %p3, align 4
  %p4 = getelementptr inbounds %struct.ST4, %struct.ST4* %B, i64 %indvars.iv, i32 3
  store i32 12, i32* %p4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}
