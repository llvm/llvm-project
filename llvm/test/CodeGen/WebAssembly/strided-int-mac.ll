; RUN: opt -mattr=+simd128 -passes=loop-vectorize %s | llc -mtriple=wasm32 -mattr=+simd128 -verify-machineinstrs -o - | FileCheck %s
; RUN: opt -mattr=+simd128 -passes=loop-vectorize -vectorizer-maximize-bandwidth %s | llc -mtriple=wasm32 -mattr=+simd128 -verify-machineinstrs -o - | FileCheck %s --check-prefix=MAX-BANDWIDTH
; RUN: opt -mattr=+simd128,+relaxed-simd -passes=loop-vectorize -vectorizer-maximize-bandwidth %s | llc -mtriple=wasm32 -mattr=+simd128,+relaxed-simd -verify-machineinstrs -o - | FileCheck %s --check-prefix=RELAXED-MAX-BANDWIDTH

target triple = "wasm32"

; CHECK-LABEL: bb2053_inner_loop:
; CHECK: loop
; CHECK: v128.load
; CHECK: i32x4.extract_lane	3
; CHECK: i32x4.extract_lane	2
; CHECK: i32x4.extract_lane	1
; CHECK: i32x4.extract_lane	0
; CHECK: v128.load8_splat	0
; CHECK: v128.load8_lane	0, 1
; CHECK: v128.load8_lane	0, 2
; CHECK: v128.load8_lane	0, 3
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: v128.load
; CHECK: i8x16.shuffle	1, 5, 9, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: v128.load8_splat	0
; CHECK: v128.load8_lane	0, 1
; CHECK: v128.load8_lane	0, 2
; CHECK: v128.load8_lane	0, 3
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i8x16.shuffle	3, 7, 11, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i8x16.shuffle	0, 4, 8, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i8x16.shuffle	2, 6, 10, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: v128.load8_splat	0
; CHECK: v128.load8_lane	0, 1
; CHECK: v128.load8_lane	0, 2
; CHECK: v128.load8_lane	0, 3
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: v128.load8_splat	0
; CHECK: v128.load8_lane	0, 1
; CHECK: v128.load8_lane	0, 2
; CHECK: v128.load8_lane	0, 3
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extract_lane	3
; MAX-BANDWIDTH: i32x4.extract_lane	2
; MAX-BANDWIDTH: i32x4.extract_lane	1
; MAX-BANDWIDTH: i32x4.extract_lane	0
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extract_lane	3
; MAX-BANDWIDTH: i32x4.extract_lane	2
; MAX-BANDWIDTH: i32x4.extract_lane	1
; MAX-BANDWIDTH: i32x4.extract_lane	0
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extract_lane	3
; MAX-BANDWIDTH: i32x4.extract_lane	2
; MAX-BANDWIDTH: i32x4.extract_lane	1
; MAX-BANDWIDTH: i32x4.extract_lane	0
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i32x4.extract_lane	3
; MAX-BANDWIDTH: i32x4.extract_lane	2
; MAX-BANDWIDTH: i32x4.extract_lane	1
; MAX-BANDWIDTH: i32x4.extract_lane	0
; MAX-BANDWIDTH: v128.load8_splat	0
; MAX-BANDWIDTH: v128.load8_lane	0, 1
; MAX-BANDWIDTH: v128.load8_lane	0, 2
; MAX-BANDWIDTH: v128.load8_lane	0, 3
; MAX-BANDWIDTH: v128.load8_lane	0, 4
; MAX-BANDWIDTH: v128.load8_lane	0, 5
; MAX-BANDWIDTH: v128.load8_lane	0, 6
; MAX-BANDWIDTH: v128.load8_lane	0, 7
; MAX-BANDWIDTH: v128.load8_lane	0, 8
; MAX-BANDWIDTH: v128.load8_lane	0, 9
; MAX-BANDWIDTH: v128.load8_lane	0, 10
; MAX-BANDWIDTH: v128.load8_lane	0, 11
; MAX-BANDWIDTH: v128.load8_lane	0, 12
; MAX-BANDWIDTH: v128.load8_lane	0, 13
; MAX-BANDWIDTH: v128.load8_lane	0, 14
; MAX-BANDWIDTH: v128.load8_lane	0, 15
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	3, 7, 11, 15, 19, 23, 27, 31, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 11, 15, 19, 23, 27, 31
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load8_splat	0
; MAX-BANDWIDTH: v128.load8_lane	0, 1
; MAX-BANDWIDTH: v128.load8_lane	0, 2
; MAX-BANDWIDTH: v128.load8_lane	0, 3
; MAX-BANDWIDTH: v128.load8_lane	0, 4
; MAX-BANDWIDTH: v128.load8_lane	0, 5
; MAX-BANDWIDTH: v128.load8_lane	0, 6
; MAX-BANDWIDTH: v128.load8_lane	0, 7
; MAX-BANDWIDTH: v128.load8_lane	0, 8
; MAX-BANDWIDTH: v128.load8_lane	0, 9
; MAX-BANDWIDTH: v128.load8_lane	0, 10
; MAX-BANDWIDTH: v128.load8_lane	0, 11
; MAX-BANDWIDTH: v128.load8_lane	0, 12
; MAX-BANDWIDTH: v128.load8_lane	0, 13
; MAX-BANDWIDTH: v128.load8_lane	0, 14
; MAX-BANDWIDTH: v128.load8_lane	0, 15
; MAX-BANDWIDTH: i8x16.shuffle	1, 5, 9, 13, 17, 21, 25, 29, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i8x16.shuffle	0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 9, 13, 17, 21, 25, 29
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i8x16.shuffle	0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 10, 14, 18, 22, 26, 30
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0
; MAX-BANDWIDTH: i8x16.shuffle	0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20, 24, 28
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load8_splat	0
; MAX-BANDWIDTH: v128.load8_lane	0, 1
; MAX-BANDWIDTH: v128.load8_lane	0, 2
; MAX-BANDWIDTH: v128.load8_lane	0, 3
; MAX-BANDWIDTH: v128.load8_lane	0, 4
; MAX-BANDWIDTH: v128.load8_lane	0, 5
; MAX-BANDWIDTH: v128.load8_lane	0, 6
; MAX-BANDWIDTH: v128.load8_lane	0, 7
; MAX-BANDWIDTH: v128.load8_lane	0, 8
; MAX-BANDWIDTH: v128.load8_lane	0, 9
; MAX-BANDWIDTH: v128.load8_lane	0, 10
; MAX-BANDWIDTH: v128.load8_lane	0, 11
; MAX-BANDWIDTH: v128.load8_lane	0, 12
; MAX-BANDWIDTH: v128.load8_lane	0, 13
; MAX-BANDWIDTH: v128.load8_lane	0, 14
; MAX-BANDWIDTH: v128.load8_lane	0, 15
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: v128.load8_splat	0
; MAX-BANDWIDTH: v128.load8_lane	0, 1
; MAX-BANDWIDTH: v128.load8_lane	0, 2
; MAX-BANDWIDTH: v128.load8_lane	0, 3
; MAX-BANDWIDTH: v128.load8_lane	0, 4
; MAX-BANDWIDTH: v128.load8_lane	0, 5
; MAX-BANDWIDTH: v128.load8_lane	0, 6
; MAX-BANDWIDTH: v128.load8_lane	0, 7
; MAX-BANDWIDTH: v128.load8_lane	0, 8
; MAX-BANDWIDTH: v128.load8_lane	0, 9
; MAX-BANDWIDTH: v128.load8_lane	0, 10
; MAX-BANDWIDTH: v128.load8_lane	0, 11
; MAX-BANDWIDTH: v128.load8_lane	0, 12
; MAX-BANDWIDTH: v128.load8_lane	0, 13
; MAX-BANDWIDTH: v128.load8_lane	0, 14
; MAX-BANDWIDTH: v128.load8_lane	0, 15
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add

; RELAXED-MAX-BANDWIDTH: loop
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	3
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	2
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	1
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	0
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	3
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	2
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	1
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	0
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	3
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	2
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	1
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	0
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	3
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	2
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	1
; RELAXED-MAX-BANDWIDTH: i32x4.extract_lane	0
; RELAXED-MAX-BANDWIDTH: v128.load8_splat	0
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 1
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 2
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 3
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 4
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 5
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 6
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 7
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 8
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 9
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 10
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 11
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 12
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 13
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 14
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 15
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	3, 7, 11, 15, 19, 23, 27, 31, 0, 0, 0, 0, 0, 0, 0, 0
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 0, 0, 0, 0, 0, 0, 0, 3, 7, 11, 15, 19, 23, 27, 31
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; RELAXED-MAX-BANDWIDTH: v128.load8_splat	0
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 1
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 2
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 3
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 4
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 5
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 6
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 7
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 8
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 9
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 10
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 11
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 12
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 13
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 14
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 15
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	1, 5, 9, 13, 17, 21, 25, 29, 0, 0, 0, 0, 0, 0, 0, 0
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 0, 0, 0, 0, 0, 0, 0, 1, 5, 9, 13, 17, 21, 25, 29
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	2, 6, 10, 14, 18, 22, 26, 30, 0, 0, 0, 0, 0, 0, 0, 0
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 0, 0, 0, 0, 0, 0, 0, 2, 6, 10, 14, 18, 22, 26, 30
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 8, 12, 16, 20, 24, 28
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: v128.load8_splat	0
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 1
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 2
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 3
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 4
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 5
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 6
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 7
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 8
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 9
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 10
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 11
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 12
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 13
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 14
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 15
; RELAXED-MAX-BANDWIDTH: v128.load8_splat	0
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 1
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 2
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 3
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 4
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 5
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 6
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 7
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 8
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 9
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 10
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 11
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 12
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 13
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 14
; RELAXED-MAX-BANDWIDTH: v128.load8_lane	0, 15
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
define hidden { i32, i32, i32, i32 } @bb2053_inner_loop(ptr nocapture %base0, ptr nocapture %base1, ptr nocapture %weights, ptr nocapture readonly %indices, i32 %len, i32 %stride, i32 %acc0, i32 %acc1, i32 %acc2, i32 %acc3) local_unnamed_addr {
entry:
  br label %bb2053.loop

bb2053.loop:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %bb2053.loop ]
  %accA = phi i32 [ %acc0, %entry ], [ %accA.sum, %bb2053.loop ]
  %accB = phi i32 [ %acc1, %entry ], [ %accB.sum, %bb2053.loop ]
  %accC = phi i32 [ %acc2, %entry ], [ %accC.sum, %bb2053.loop ]
  %accD = phi i32 [ %acc3, %entry ], [ %accD.sum, %bb2053.loop ]
  %wptr = phi ptr [ %weights, %entry ], [ %wptr.next, %bb2053.loop ]
  %idx.ptr = getelementptr inbounds nuw i32, ptr %indices, i32 %idx
  %idx.val = load i32, ptr %idx.ptr, align 4
  %lhs0.ptr = getelementptr inbounds i8, ptr %base0, i32 %idx.val
  %rhs0.ptr = getelementptr inbounds i8, ptr %base1, i32 %idx.val
  %lhs0 = load i8, ptr %lhs0.ptr, align 1
  %lhs0.sext = sext i8 %lhs0 to i32
  %w0 = load i8, ptr %wptr, align 1
  %w0.sext = sext i8 %w0 to i32
  %mul0 = mul nsw i32 %w0.sext, %lhs0.sext
  %accA.next = add nsw i32 %mul0, %accA
  %w1.ptr = getelementptr inbounds nuw i8, ptr %wptr, i32 1
  %w1 = load i8, ptr %w1.ptr, align 1
  %w1.sext = sext i8 %w1 to i32
  %mul1 = mul nsw i32 %w1.sext, %lhs0.sext
  %accC.next = add nsw i32 %mul1, %accC
  %lhs1.ptr = getelementptr inbounds nuw i8, ptr %lhs0.ptr, i32 %stride
  %lhs1 = load i8, ptr %lhs1.ptr, align 1
  %lhs1.sext = sext i8 %lhs1 to i32
  %w2.ptr = getelementptr inbounds nuw i8, ptr %wptr, i32 2
  %w2 = load i8, ptr %w2.ptr, align 1
  %w2.sext = sext i8 %w2 to i32
  %mul2 = mul nsw i32 %w2.sext, %lhs1.sext
  %accA.sum = add nsw i32 %accA.next, %mul2
  %w3.ptr = getelementptr inbounds nuw i8, ptr %wptr, i32 3
  %w3 = load i8, ptr %w3.ptr, align 1
  %w3.sext = sext i8 %w3 to i32
  %mul3 = mul nsw i32 %w3.sext, %lhs1.sext
  %accC.sum = add nsw i32 %accC.next, %mul3
  %rhs0 = load i8, ptr %rhs0.ptr, align 1
  %rhs0.sext = sext i8 %rhs0 to i32
  %mul4 = mul nsw i32 %rhs0.sext, %w0.sext
  %accB.next = add nsw i32 %mul4, %accB
  %mul5 = mul nsw i32 %rhs0.sext, %w1.sext
  %accD.next = add nsw i32 %mul5, %accD
  %rhs1.ptr = getelementptr inbounds nuw i8, ptr %rhs0.ptr, i32 %stride
  %rhs1 = load i8, ptr %rhs1.ptr, align 1
  %rhs1.sext = sext i8 %rhs1 to i32
  %mul6 = mul nsw i32 %rhs1.sext, %w2.sext
  %accB.sum = add nsw i32 %accB.next, %mul6
  %mul7 = mul nsw i32 %rhs1.sext, %w3.sext
  %accD.sum = add nsw i32 %accD.next, %mul7
  %wptr.next = getelementptr inbounds nuw i8, ptr %wptr, i32 4
  %idx.next = add nuw nsw i32 %idx, 1
  %exit = icmp eq i32 %idx.next, %len
  br i1 %exit, label %bb2053.exit, label %bb2053.loop

bb2053.exit:
  %res0 = insertvalue { i32, i32, i32, i32 } poison, i32 %accA.sum, 0
  %res1 = insertvalue { i32, i32, i32, i32 } %res0, i32 %accB.sum, 1
  %res2 = insertvalue { i32, i32, i32, i32 } %res1, i32 %accC.sum, 2
  %res3 = insertvalue { i32, i32, i32, i32 } %res2, i32 %accD.sum, 3
  ret { i32, i32, i32, i32 } %res3
}

; CHECK-LABEL: bb41_inner_loop:
; CHECK: loop
; CHECK: v128.load64_zero
; CHECK: i8x16.shuffle	1, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: v128.load64_zero
; CHECK: i8x16.shuffle	1, 3, 5, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i8x16.shuffle	0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i8x16.shuffle	0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
; CHECK: i16x8.extend_low_i8x16_s
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i16x8.extmul_low_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i16x8.extmul_high_i8x16_s
; MAX-BANDWIDTH: i32x4.extadd_pairwise_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.add

; RELAXED-MAX-BANDWIDTH: loop
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
; RELAXED-MAX-BANDWIDTH: i32x4.relaxed_dot_i8x16_i7x16_add_s
define hidden { i32, i32, i32, i32 } @bb41_inner_loop(ptr nocapture %lhs, ptr nocapture %rhs, i32 %len, i32 %acc00, i32 %acc01, i32 %acc10, i32 %acc11) local_unnamed_addr {
entry:
  br label %bb41

bb41:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %bb41 ]
  %lhs.ptr = phi ptr [ %lhs, %entry ], [ %lhs.next, %bb41 ]
  %rhs.ptr = phi ptr [ %rhs, %entry ], [ %rhs.next, %bb41 ]
  %acc00.phi = phi i32 [ %acc00, %entry ], [ %acc00.next, %bb41 ]
  %acc01.phi = phi i32 [ %acc01, %entry ], [ %acc01.next, %bb41 ]
  %acc10.phi = phi i32 [ %acc10, %entry ], [ %acc10.next, %bb41 ]
  %acc11.phi = phi i32 [ %acc11, %entry ], [ %acc11.next, %bb41 ]
  %lhs0 = load i8, ptr %lhs.ptr, align 1
  %lhs0.sext = sext i8 %lhs0 to i32
  %rhs0 = load i8, ptr %rhs.ptr, align 1
  %rhs0.sext = sext i8 %rhs0 to i32
  %mul00 = mul nsw i32 %rhs0.sext, %lhs0.sext
  %acc00.next = add nsw i32 %mul00, %acc00.phi
  %rhs1.ptr = getelementptr inbounds nuw i8, ptr %rhs.ptr, i32 1
  %rhs1 = load i8, ptr %rhs1.ptr, align 1
  %rhs1.sext = sext i8 %rhs1 to i32
  %mul01 = mul nsw i32 %rhs1.sext, %lhs0.sext
  %acc01.next = add nsw i32 %mul01, %acc01.phi
  %lhs1.ptr = getelementptr inbounds nuw i8, ptr %lhs.ptr, i32 1
  %lhs1 = load i8, ptr %lhs1.ptr, align 1
  %lhs1.sext = sext i8 %lhs1 to i32
  %mul10 = mul nsw i32 %lhs1.sext, %rhs0.sext
  %acc10.next = add nsw i32 %mul10, %acc10.phi
  %mul11 = mul nsw i32 %lhs1.sext, %rhs1.sext
  %acc11.next = add nsw i32 %mul11, %acc11.phi
  %lhs.next = getelementptr inbounds nuw i8, ptr %lhs.ptr, i32 2
  %rhs.next = getelementptr inbounds nuw i8, ptr %rhs.ptr, i32 2
  %idx.next = add nuw nsw i32 %idx, 1
  %exit = icmp eq i32 %idx.next, %len
  br i1 %exit, label %bb41.exit, label %bb41

bb41.exit:
  %res0 = insertvalue { i32, i32, i32, i32 } poison, i32 %acc00.next, 0
  %res1 = insertvalue { i32, i32, i32, i32 } %res0, i32 %acc01.next, 1
  %res2 = insertvalue { i32, i32, i32, i32 } %res1, i32 %acc10.next, 2
  %res3 = insertvalue { i32, i32, i32, i32 } %res2, i32 %acc11.next, 3
  ret { i32, i32, i32, i32 } %res3
}

; CHECK-LABEL: bb41_inner_loop_i16:
; CHECK: loop
; CHECK: v128.load
; CHECK: v128.load
; CHECK: i8x16.shuffle	4, 5, 12, 13, 20, 21, 28, 29, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: v128.load
; CHECK: v128.load
; CHECK: i8x16.shuffle	4, 5, 12, 13, 20, 21, 28, 29, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i8x16.shuffle	0, 1, 8, 9, 16, 17, 24, 25, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i8x16.shuffle	0, 1, 8, 9, 16, 17, 24, 25, 0, 1, 0, 1, 0, 1, 0, 1
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add
; CHECK: i32x4.extmul_low_i16x8_s
; CHECK: i32x4.add

; MAX-BANDWIDTH: loop
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 12, 13, 20, 21, 28, 29, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 0, 1, 0, 1, 0, 1, 4, 5, 12, 13, 20, 21, 28, 29
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	4, 5, 12, 13, 20, 21, 28, 29, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: v128.load
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 0, 1, 0, 1, 0, 1, 4, 5, 12, 13, 20, 21, 28, 29
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; MAX-BANDWIDTH: i32x4.dot_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 8, 9, 16, 17, 24, 25, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 16, 17, 24, 25
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; MAX-BANDWIDTH: i32x4.dot_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 8, 9, 16, 17, 24, 25, 0, 1, 0, 1, 0, 1, 0, 1
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 16, 17, 24, 25
; MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; MAX-BANDWIDTH: i32x4.dot_i16x8_s
; MAX-BANDWIDTH: i32x4.add
; MAX-BANDWIDTH: i32x4.dot_i16x8_s
; MAX-BANDWIDTH: i32x4.add

; RELAXED-MAX-BANDWIDTH: loop
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	4, 5, 12, 13, 20, 21, 28, 29, 0, 1, 0, 1, 0, 1, 0, 1
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 0, 1, 0, 1, 0, 1, 4, 5, 12, 13, 20, 21, 28, 29
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	4, 5, 12, 13, 20, 21, 28, 29, 0, 1, 0, 1, 0, 1, 0, 1
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: v128.load
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 0, 1, 0, 1, 0, 1, 4, 5, 12, 13, 20, 21, 28, 29
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; RELAXED-MAX-BANDWIDTH: i32x4.dot_i16x8_s
; RELAXED-MAX-BANDWIDTH: i32x4.add
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 8, 9, 16, 17, 24, 25, 0, 1, 0, 1, 0, 1, 0, 1
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 16, 17, 24, 25
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; RELAXED-MAX-BANDWIDTH: i32x4.dot_i16x8_s
; RELAXED-MAX-BANDWIDTH: i32x4.add
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 8, 9, 16, 17, 24, 25, 0, 1, 0, 1, 0, 1, 0, 1
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 8, 9, 16, 17, 24, 25
; RELAXED-MAX-BANDWIDTH: i8x16.shuffle	0, 1, 2, 3, 4, 5, 6, 7, 24, 25, 26, 27, 28, 29, 30, 31
; RELAXED-MAX-BANDWIDTH: i32x4.dot_i16x8_s
; RELAXED-MAX-BANDWIDTH: i32x4.add
; RELAXED-MAX-BANDWIDTH: i32x4.dot_i16x8_s
; RELAXED-MAX-BANDWIDTH: i32x4.add
define hidden { i32, i32, i32, i32 } @bb41_inner_loop_i16(ptr nocapture %lhs, ptr nocapture %rhs, i32 %len, i32 %acc00, i32 %acc01, i32 %acc10, i32 %acc11) local_unnamed_addr {
entry:
  br label %bb41

bb41:
  %idx = phi i32 [ 0, %entry ], [ %idx.next, %bb41 ]
  %lhs.ptr = phi ptr [ %lhs, %entry ], [ %lhs.next, %bb41 ]
  %rhs.ptr = phi ptr [ %rhs, %entry ], [ %rhs.next, %bb41 ]
  %acc00.phi = phi i32 [ %acc00, %entry ], [ %acc00.next, %bb41 ]
  %acc01.phi = phi i32 [ %acc01, %entry ], [ %acc01.next, %bb41 ]
  %acc10.phi = phi i32 [ %acc10, %entry ], [ %acc10.next, %bb41 ]
  %acc11.phi = phi i32 [ %acc11, %entry ], [ %acc11.next, %bb41 ]
  %lhs0 = load i16, ptr %lhs.ptr, align 2
  %lhs0.sext = sext i16 %lhs0 to i32
  %rhs0 = load i16, ptr %rhs.ptr, align 2
  %rhs0.sext = sext i16 %rhs0 to i32
  %mul00 = mul nsw i32 %rhs0.sext, %lhs0.sext
  %acc00.next = add nsw i32 %mul00, %acc00.phi
  %rhs1.ptr = getelementptr inbounds nuw i16, ptr %rhs.ptr, i32 2
  %rhs1 = load i16, ptr %rhs1.ptr, align 2
  %rhs1.sext = sext i16 %rhs1 to i32
  %mul01 = mul nsw i32 %rhs1.sext, %lhs0.sext
  %acc01.next = add nsw i32 %mul01, %acc01.phi
  %lhs1.ptr = getelementptr inbounds nuw i16, ptr %lhs.ptr, i32 2
  %lhs1 = load i16, ptr %lhs1.ptr, align 2
  %lhs1.sext = sext i16 %lhs1 to i32
  %mul10 = mul nsw i32 %lhs1.sext, %rhs0.sext
  %acc10.next = add nsw i32 %mul10, %acc10.phi
  %mul11 = mul nsw i32 %lhs1.sext, %rhs1.sext
  %acc11.next = add nsw i32 %mul11, %acc11.phi
  %lhs.next = getelementptr inbounds nuw i16, ptr %lhs.ptr, i32 4
  %rhs.next = getelementptr inbounds nuw i16, ptr %rhs.ptr, i32 4
  %idx.next = add nuw nsw i32 %idx, 1
  %exit = icmp eq i32 %idx.next, %len
  br i1 %exit, label %bb41.exit, label %bb41

bb41.exit:
  %res0 = insertvalue { i32, i32, i32, i32 } poison, i32 %acc00.next, 0
  %res1 = insertvalue { i32, i32, i32, i32 } %res0, i32 %acc01.next, 1
  %res2 = insertvalue { i32, i32, i32, i32 } %res1, i32 %acc10.next, 2
  %res3 = insertvalue { i32, i32, i32, i32 } %res2, i32 %acc11.next, 3
  ret { i32, i32, i32, i32 } %res3
}
