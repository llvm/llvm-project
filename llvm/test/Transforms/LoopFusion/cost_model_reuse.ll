; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-max-reuse-distance=64 -disable-output -pass-remarks=loop-fusion -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s --check-prefix=DEFAULT
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-max-reuse-distance=64 -loop-fusion-min-reused-values=2 -disable-output -pass-remarks=loop-fusion -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s --check-prefix=MIN2
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-max-reuse-distance=64 -loop-fusion-min-reused-values=3 -disable-output -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s --check-prefix=MIN3
; RUN: opt -passes=loop-simplify,loop-fusion -disable-output -pass-remarks=loop-fusion -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s --check-prefix=FALLBACK
; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-max-reuse-distance=32 -disable-output -pass-remarks=loop-fusion -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s --check-prefix=DIST32

; DEFAULT: [no_cross_loop_reuse]{{.*}}found 0 cross-loop reused values; configured minimum is 1
; DEFAULT: [exact_and_nearby_rar]{{.*}}Loops fused
; DEFAULT: [one_raw_producer_multiple_reads]{{.*}}Loops fused
; DEFAULT: [two_reads_two_spatial_matches]{{.*}}Loops fused
; DEFAULT: [two_raw_producers]{{.*}}Loops fused
; The distance must be strictly within the configured limit. 7 i32 elements
; are 28 bytes and count for both limits. 8 elements are exactly 32 bytes.
; 16 elements are exactly 64 bytes.
; DEFAULT: [offset_7]{{.*}}Loops fused
; DEFAULT: [offset_8]{{.*}}Loops fused
; DEFAULT: [offset_16]{{.*}}found 0 cross-loop reused values

; MIN2: [no_cross_loop_reuse]{{.*}}found 0 cross-loop reused values; configured minimum is 2
; MIN2: [exact_and_nearby_rar]{{.*}}Loops fused
; MIN2: [one_raw_producer_multiple_reads]{{.*}}found 1 cross-loop reused values; configured minimum is 2
; MIN2: [two_reads_two_spatial_matches]{{.*}}Loops fused
; MIN2: [two_raw_producers]{{.*}}Loops fused

; MIN3: [exact_and_nearby_rar]{{.*}}found 2 cross-loop reused values; configured minimum is 3
; MIN3: [two_reads_two_spatial_matches]{{.*}}found 2 cross-loop reused values; configured minimum is 3
; MIN3: [two_raw_producers]{{.*}}found 2 cross-loop reused values; configured minimum is 3

; FALLBACK omits the option, so it uses the default of 64 and matches DEFAULT.
; FALLBACK: [offset_7]{{.*}}Loops fused
; FALLBACK: [offset_8]{{.*}}Loops fused
; FALLBACK: [offset_16]{{.*}}found 0 cross-loop reused values
; DIST32: [offset_7]{{.*}}Loops fused
; DIST32: [offset_8]{{.*}}found 0 cross-loop reused values
; DIST32: [offset_16]{{.*}}found 0 cross-loop reused values

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

; The exact read and the constant-distance read are two reused values.
define void @exact_and_nearby_rar(
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

; Both loop-1 loads are within the reuse limit of both loop-2 reads, so each
; loop-1 load counts. The two reads contribute two reused values.
define void @two_reads_two_spatial_matches(ptr noalias %src, ptr noalias %dst1,
                                         ptr noalias %dst2, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1.latch ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %next.i1 = add nuw nsw i64 %i1, 1
  %next.gep1 = getelementptr inbounds i32, ptr %src, i64 %next.i1
  %next.value1 = load i32, ptr %next.gep1, align 4
  %sum1 = add i32 %value1, %next.value1
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %sum1, ptr %dst.gep1, align 4
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
  %next.i2 = add nuw nsw i64 %i2, 1
  %next.gep2 = getelementptr inbounds i32, ptr %src, i64 %next.i2
  %next.value2 = load i32, ptr %next.gep2, align 4
  %sum2 = add i32 %value2, %next.value2
  %dst.gep2 = getelementptr inbounds i32, ptr %dst2, i64 %i2
  store i32 %sum2, ptr %dst.gep2, align 4
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

; 7 i32 elements are 28 bytes, within both a 32-byte and a 64-byte limit.
define void @offset_7(ptr noalias %src, ptr noalias %dst1, ptr noalias %dst2,
                      i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1 ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %value1, ptr %dst.gep1, align 4
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2 ]
  %shifted = add nuw nsw i64 %i2, 7
  %src.gep2 = getelementptr inbounds i32, ptr %src, i64 %shifted
  %value2 = load i32, ptr %src.gep2, align 4
  %dst.gep2 = getelementptr inbounds i32, ptr %dst2, i64 %i2
  store i32 %value2, ptr %dst.gep2, align 4
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

; 8 i32 elements are 32 bytes: within a 64-byte limit, and exactly the 32-byte
; limit, so the 32-byte run rejects it.
define void @offset_8(ptr noalias %src, ptr noalias %dst1, ptr noalias %dst2,
                      i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1 ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %value1, ptr %dst.gep1, align 4
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2 ]
  %shifted = add nuw nsw i64 %i2, 8
  %src.gep2 = getelementptr inbounds i32, ptr %src, i64 %shifted
  %value2 = load i32, ptr %src.gep2, align 4
  %dst.gep2 = getelementptr inbounds i32, ptr %dst2, i64 %i2
  store i32 %value2, ptr %dst.gep2, align 4
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

; 16 i32 elements are 64 bytes, exactly the default limit, so they do not count.
define void @offset_16(ptr noalias %src, ptr noalias %dst1, ptr noalias %dst2,
                       i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1 ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %value1, ptr %dst.gep1, align 4
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2 ]
  %shifted = add nuw nsw i64 %i2, 16
  %src.gep2 = getelementptr inbounds i32, ptr %src, i64 %shifted
  %value2 = load i32, ptr %src.gep2, align 4
  %dst.gep2 = getelementptr inbounds i32, ptr %dst2, i64 %i2
  store i32 %value2, ptr %dst.gep2, align 4
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}
