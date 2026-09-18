; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-cost-model -disable-output -pass-remarks=loop-fusion -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-cost-model -loop-fusion-min-reused-values=2 -disable-output -pass-remarks=loop-fusion -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s --check-prefix=MIN2
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-cost-model -loop-fusion-min-reused-values=3 -disable-output -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s --check-prefix=MIN3

; DEFAULT: [no_cross_loop_reuse]{{.*}}found 0 cross-loop reused values; configured minimum is 1
; DEFAULT: [exact_rar_with_unmatched_read]{{.*}}Loops fused
; DEFAULT: [one_raw_producer_multiple_reads]{{.*}}Loops fused
; DEFAULT: [two_raw_producers]{{.*}}Loops fused

; MIN2: [no_cross_loop_reuse]{{.*}}found 0 cross-loop reused values; configured minimum is 2
; MIN2: [exact_rar_with_unmatched_read]{{.*}}found 1 cross-loop reused values; configured minimum is 2
; MIN2: [one_raw_producer_multiple_reads]{{.*}}found 1 cross-loop reused values; configured minimum is 2
; MIN2: [two_raw_producers]{{.*}}Loops fused

; MIN3: [two_raw_producers]{{.*}}found 2 cross-loop reused values; configured minimum is 3

; Independent stores have no cross-loop reuse and are rejected by default.
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

; One exact affine RAR match is sufficient. The additional shifted read does
; not match and must not increase the reused-value count.
define void @exact_rar_with_unmatched_read(
    ptr noalias %src, ptr noalias %other, ptr noalias %a, ptr noalias %b,
    i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %other.gep1 = getelementptr i32, ptr %other, i64 %i1
  %other.value1 = load i32, ptr %other.gep1, align 4
  %sum1 = add i32 %value1, %other.value1
  %a.gep = getelementptr inbounds i32, ptr %a, i64 %i1
  store i32 %sum1, ptr %a.gep, align 4
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
  %other.index2 = add i64 %i2, 1
  %other.gep2 = getelementptr i32, ptr %other, i64 %other.index2
  %other.value2 = load i32, ptr %other.gep2, align 4
  %sum2 = add i32 %value2, %other.value2
  %b.gep = getelementptr inbounds i32, ptr %b, i64 %i2
  store i32 %sum2, ptr %b.gep, align 4
  br label %loop2.latch

loop2.latch:
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

; Multiple consumer loads reached by the same producer count as one distinct
; reused value.
define void @one_raw_producer_multiple_reads(ptr noalias %a, ptr noalias %b,
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

; Distinct producer stores count independently.
define void @two_raw_producers(ptr noalias %a, ptr noalias %b,
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
