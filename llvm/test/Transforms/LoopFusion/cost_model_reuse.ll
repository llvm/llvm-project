; REQUIRES: asserts
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-cost-model \
; RUN:   -disable-output -stats -pass-remarks-missed=loop-fusion < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MIN1
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-cost-model \
; RUN:   -loop-fusion-min-reused-values=2 \
; RUN:   -disable-output -stats -pass-remarks-missed=loop-fusion < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MIN2
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-cost-model \
; RUN:   -loop-fusion-min-reused-values=3 \
; RUN:   -disable-output -pass-remarks-missed=loop-fusion < %s 2>&1 | \
; RUN:   FileCheck %s --check-prefix=MIN3

; MIN1: [no_cross_loop_reuse]{{.*}}found 0 cross-loop reused values; configured minimum is 1
; MIN1: [shared_reads]{{.*}}found 0 cross-loop reused values; configured minimum is 1
; MIN1: 2 loop-fusion - Loops fused
; MIN1: 2 loop-fusion - Fusion has insufficient cross-loop reused values

; MIN2: [no_cross_loop_reuse]{{.*}}found 0 cross-loop reused values; configured minimum is 2
; MIN2: [shared_reads]{{.*}}found 0 cross-loop reused values; configured minimum is 2
; MIN2: [one_producer_multiple_reads]{{.*}}found 1 cross-loop reused values; configured minimum is 2
; MIN2: 1 loop-fusion - Loops fused
; MIN2: 3 loop-fusion - Fusion has insufficient cross-loop reused values

; MIN3: [two_distinct_producers]{{.*}}found 2 cross-loop reused values; configured minimum is 3

define void @no_cross_loop_reuse(ptr noalias %a, ptr noalias %b, i32 %x,
                                 i32 %y, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %a.gep = getelementptr inbounds i32, ptr %a, i64 %i1
  store i32 %x, ptr %a.gep, align 4
  br label %loop1.latch

loop1.latch:
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2.latch ]
  %b.gep = getelementptr inbounds i32, ptr %b, i64 %i2
  store i32 %y, ptr %b.gep, align 4
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

define void @shared_reads(ptr noalias %src, ptr noalias %a, ptr noalias %b,
                          i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %a.gep = getelementptr inbounds i32, ptr %a, i64 %i1
  store i32 %value1, ptr %a.gep, align 4
  br label %loop1.latch

loop1.latch:
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2.latch ]
  %src.gep2 = getelementptr inbounds i32, ptr %src, i64 %i2
  %value2 = load i32, ptr %src.gep2, align 4
  %b.gep = getelementptr inbounds i32, ptr %b, i64 %i2
  store i32 %value2, ptr %b.gep, align 4
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

define void @one_producer_multiple_reads(ptr noalias %a, ptr noalias %b,
                                         i32 %x, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %a.gep1 = getelementptr inbounds i32, ptr %a, i64 %i1
  store i32 %x, ptr %a.gep1, align 4
  br label %loop1.latch

loop1.latch:
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2.latch ]
  %a.gep2 = getelementptr inbounds i32, ptr %a, i64 %i2
  %value1 = load i32, ptr %a.gep2, align 4
  %value2 = load i32, ptr %a.gep2, align 4
  %sum = add i32 %value1, %value2
  %b.gep = getelementptr inbounds i32, ptr %b, i64 %i2
  store i32 %sum, ptr %b.gep, align 4
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

define void @two_distinct_producers(ptr noalias %a, ptr noalias %b,
                                    ptr noalias %c, i32 %x, i32 %y, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %a.gep1 = getelementptr inbounds i32, ptr %a, i64 %i1
  store i32 %x, ptr %a.gep1, align 4
  %b.gep1 = getelementptr inbounds i32, ptr %b, i64 %i1
  store i32 %y, ptr %b.gep1, align 4
  br label %loop1.latch

loop1.latch:
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2.latch ]
  %a.gep2 = getelementptr inbounds i32, ptr %a, i64 %i2
  %value1 = load i32, ptr %a.gep2, align 4
  %b.gep2 = getelementptr inbounds i32, ptr %b, i64 %i2
  %value2 = load i32, ptr %b.gep2, align 4
  %sum = add i32 %value1, %value2
  %c.gep = getelementptr inbounds i32, ptr %c, i64 %i2
  store i32 %sum, ptr %c.gep, align 4
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}
