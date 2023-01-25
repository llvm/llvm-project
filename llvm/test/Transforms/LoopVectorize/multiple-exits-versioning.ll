; RUN: opt -passes=loop-vectorize -force-vector-width=2 -S %s | FileCheck %s

; Test cases to make sure LV & loop versioning can handle loops with
; multiple exiting branches.

; Multiple branches exiting the loop to a unique exit block. The loop should
; be vectorized with versioning.
define void @multiple_exits_unique_exit_block(ptr %A, ptr %B, i64 %N) {
; CHECK-LABEL: @multiple_exits_unique_exit_block
; CHECK:       vector.memcheck:
; CHECK-LABEL: vector.body:
; CHECK:         %wide.load = load <2 x i32>, ptr {{.*}}, align 4
; CHECK:         store <2 x i32> %wide.load, ptr {{.*}}, align 4
; CHECK:         br
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %cond.0 = icmp eq i64 %iv, %N
  br i1 %cond.0, label %exit, label %for.body

for.body:
  %A.gep = getelementptr inbounds i32, ptr %A, i64 %iv
  %lv = load i32, ptr %A.gep, align 4
  %B.gep = getelementptr inbounds i32, ptr %B, i64 %iv
  store i32 %lv, ptr %B.gep, align 4
  %iv.next = add nuw i64 %iv, 1
  %cond.1 = icmp ult i64 %iv.next, 1000
  br i1 %cond.1, label %loop.header, label %exit

exit:
  ret void
}


; Multiple branches exiting the loop to different blocks. Currently this is not supported.
define i32 @multiple_exits_multiple_exit_blocks(ptr %A, ptr %B, i64 %N) {
; CHECK-LABEL: @multiple_exits_multiple_exit_blocks
; CHECK-NEXT:    entry:
; CHECK:           br label %loop.header
; CHECK-NOT:      <2 x i32>
;
entry:
  br label %loop.header

loop.header:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %cond.0 = icmp eq i64 %iv, %N
  br i1 %cond.0, label %exit.0, label %for.body

for.body:
  %A.gep = getelementptr inbounds i32, ptr %A, i64 %iv
  %lv = load i32, ptr %A.gep, align 4
  %B.gep = getelementptr inbounds i32, ptr %B, i64 %iv
  store i32 %lv, ptr %B.gep, align 4
  %iv.next = add nuw i64 %iv, 1
  %cond.1 = icmp ult i64 %iv.next, 1000
  br i1 %cond.1, label %loop.header, label %exit.1

exit.0:
  ret i32 1

exit.1:
  ret i32 2
}
