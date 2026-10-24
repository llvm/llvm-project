; RUN: opt < %s -passes=slp-vectorizer -S -mtriple=x86_64-- -mcpu=znver5 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,STLF-ON
; RUN: opt < %s -passes=slp-vectorizer -slp-store-load-forward-check=false -S \
; RUN:   -mtriple=x86_64-- -mcpu=znver5 \
; RUN:   | FileCheck %s --check-prefixes=CHECK,STLF-OFF

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Store and load step by -4. A 4-byte <4 x i8> store and a 16-byte <4 x i32>
; load sit 4 bytes apart and the load runs first, so this iteration forwards
; nothing. The next iteration's load starts 4 bytes lower and is 16 bytes
; wide, so it re-reads the previous store: a real hazard that the negative
; stride must not hide.
define void @negative_stride_loop_carried(ptr noalias %A, i64 %n) {
; CHECK-LABEL: define void @negative_stride_loop_carried(
; STLF-ON:       load i32
; STLF-ON-NOT:   load <4 x i32>
; STLF-OFF:      load <4 x i32>
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 64, %entry ], [ %i.next, %for.body ]
  %bm4 = add i64 %i, -4
  %bp4 = add i64 %i, 4
  %bp8 = add i64 %i, 8
  %p0 = getelementptr inbounds i8, ptr %A, i64 %bm4
  %p1 = getelementptr inbounds i8, ptr %A, i64 %i
  %p2 = getelementptr inbounds i8, ptr %A, i64 %bp4
  %p3 = getelementptr inbounds i8, ptr %A, i64 %bp8
  %l0 = load i32, ptr %p0, align 4
  %l1 = load i32, ptr %p1, align 4
  %l2 = load i32, ptr %p2, align 4
  %l3 = load i32, ptr %p3, align 4
  %a0 = add i32 %l0, 1
  %a1 = add i32 %l1, 1
  %a2 = add i32 %l2, 1
  %a3 = add i32 %l3, 1
  %t0 = trunc i32 %a0 to i8
  %t1 = trunc i32 %a1 to i8
  %t2 = trunc i32 %a2 to i8
  %t3 = trunc i32 %a3 to i8
  %i1 = add i64 %i, 1
  %i2 = add i64 %i, 2
  %i3 = add i64 %i, 3
  %s0 = getelementptr inbounds i8, ptr %A, i64 %i
  %s1 = getelementptr inbounds i8, ptr %A, i64 %i1
  %s2 = getelementptr inbounds i8, ptr %A, i64 %i2
  %s3 = getelementptr inbounds i8, ptr %A, i64 %i3
  store i8 %t0, ptr %s0, align 1
  store i8 %t1, ptr %s1, align 1
  store i8 %t2, ptr %s2, align 1
  store i8 %t3, ptr %s3, align 1
  %i.next = add nsw i64 %i, -4
  %cmp = icmp sgt i64 %i.next, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; The widened store is in the loop header and the overlapping widened load
; is in a block it dominates, so the load reads bytes this iteration just
; stored (distance 4, load width 16). The stride (32) keeps later iterations
; apart, so only the same-iteration overlap is a hazard.
define void @store_in_dominating_block(ptr noalias %A, i64 %n, i1 %c) {
; CHECK-LABEL: define void @store_in_dominating_block(
; STLF-ON:       load i32
; STLF-ON-NOT:   load <4 x i32>
; STLF-OFF:      load <4 x i32>
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 8, %entry ], [ %i.next, %latch ]
  %s0 = getelementptr inbounds i32, ptr %A, i64 %i
  %si1 = add i64 %i, 1
  %si2 = add i64 %i, 2
  %si3 = add i64 %i, 3
  %s1 = getelementptr inbounds i32, ptr %A, i64 %si1
  %s2 = getelementptr inbounds i32, ptr %A, i64 %si2
  %s3 = getelementptr inbounds i32, ptr %A, i64 %si3
  store i32 1, ptr %s0, align 4
  store i32 2, ptr %s1, align 4
  store i32 3, ptr %s2, align 4
  store i32 4, ptr %s3, align 4
  br i1 %c, label %loadbb, label %latch

loadbb:
  %bm1 = add i64 %i, -1
  %bp1 = add i64 %i, 1
  %bp2 = add i64 %i, 2
  %p0 = getelementptr inbounds i32, ptr %A, i64 %bm1
  %p1 = getelementptr inbounds i32, ptr %A, i64 %i
  %p2 = getelementptr inbounds i32, ptr %A, i64 %bp1
  %p3 = getelementptr inbounds i32, ptr %A, i64 %bp2
  %l0 = load i32, ptr %p0, align 4
  %l1 = load i32, ptr %p1, align 4
  %l2 = load i32, ptr %p2, align 4
  %l3 = load i32, ptr %p3, align 4
  %d0 = getelementptr inbounds i32, ptr %A, i64 100
  %d1 = getelementptr inbounds i32, ptr %A, i64 101
  %d2 = getelementptr inbounds i32, ptr %A, i64 102
  %d3 = getelementptr inbounds i32, ptr %A, i64 103
  store i32 %l0, ptr %d0, align 4
  store i32 %l1, ptr %d1, align 4
  store i32 %l2, ptr %d2, align 4
  store i32 %l3, ptr %d3, align 4
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 8
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; Same overlap as above, but the load and the store are on opposite sides of
; a branch, so neither dominates the other and they never run in the same
; iteration. There is no hazard, so the load must still widen.
define void @load_and_store_in_sibling_blocks(ptr noalias %A, i64 %n, i1 %c) {
; CHECK-LABEL: define void @load_and_store_in_sibling_blocks(
; CHECK:         load <4 x i32>
entry:
  br label %for.body

for.body:
  %i = phi i64 [ 8, %entry ], [ %i.next, %latch ]
  br i1 %c, label %loadbb, label %storebb

loadbb:
  %bm1 = add i64 %i, -1
  %bp1 = add i64 %i, 1
  %bp2 = add i64 %i, 2
  %p0 = getelementptr inbounds i32, ptr %A, i64 %bm1
  %p1 = getelementptr inbounds i32, ptr %A, i64 %i
  %p2 = getelementptr inbounds i32, ptr %A, i64 %bp1
  %p3 = getelementptr inbounds i32, ptr %A, i64 %bp2
  %l0 = load i32, ptr %p0, align 4
  %l1 = load i32, ptr %p1, align 4
  %l2 = load i32, ptr %p2, align 4
  %l3 = load i32, ptr %p3, align 4
  %d0 = getelementptr inbounds i32, ptr %A, i64 100
  %d1 = getelementptr inbounds i32, ptr %A, i64 101
  %d2 = getelementptr inbounds i32, ptr %A, i64 102
  %d3 = getelementptr inbounds i32, ptr %A, i64 103
  store i32 %l0, ptr %d0, align 4
  store i32 %l1, ptr %d1, align 4
  store i32 %l2, ptr %d2, align 4
  store i32 %l3, ptr %d3, align 4
  br label %latch

storebb:
  %s0 = getelementptr inbounds i32, ptr %A, i64 %i
  %si1 = add i64 %i, 1
  %si2 = add i64 %i, 2
  %si3 = add i64 %i, 3
  %s1 = getelementptr inbounds i32, ptr %A, i64 %si1
  %s2 = getelementptr inbounds i32, ptr %A, i64 %si2
  %s3 = getelementptr inbounds i32, ptr %A, i64 %si3
  store i32 1, ptr %s0, align 4
  store i32 2, ptr %s1, align 4
  store i32 3, ptr %s2, align 4
  store i32 4, ptr %s3, align 4
  br label %latch

latch:
  %i.next = add nuw nsw i64 %i, 8
  %cmp = icmp slt i64 %i.next, %n
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}
