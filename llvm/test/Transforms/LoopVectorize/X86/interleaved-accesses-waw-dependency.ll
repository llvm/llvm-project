; RUN: opt < %s -passes=loop-vectorize -force-vector-width=4 -force-vector-interleave=2 -debug-only=vectorutils -disable-output -enable-interleaved-mem-accesses=true 2>&1 | FileCheck %s
; REQUIRES: asserts
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; PR40291
; The loop does the following operation 3 times:
; 1. Load x from memory;
; 2. Store (x + 1) to this memory;
; 3. if (x < 1), store 0 to this memory.

; When scalar version stores 0 in all locations, the vector version should do
; the same thing. However, with interleaving it does not honour the WAW dependency between
; store 0 and store (x + 1) to the same memory.
; For now, we identify such unsafe dependency and disable adding the
; store into the interleaved group.
; In this test case, because we disable adding store into i32* %storeaddr12 and
; storeaddr22, we create interleaved groups with gaps and
; disable that interleaved group. So, we are only left with valid interleaved
; groups.




; CHECK:      LV: Analyzing interleaved accesses...
; CHECK:      LV: Creating an interleave group with:  store i32 %tmp34, i32* %storeaddr32, align 4
; CHECK-NEXT: LV: Inserted:  store i32 %tmp24, i32* %storeaddr22, align 4
; CHECK-NEXT:     into the interleave group with  store i32 %tmp34, i32* %storeaddr32, align 4
; CHECK-NEXT: LV: Inserted:  store i32 %tmp14, i32* %storeaddr12, align 4
; CHECK-NEXT:     into the interleave group with  store i32 %tmp34, i32* %storeaddr32, align 4
; CHECK:      LV: Invalidated store group due to dependence between   store i32 %tmp24, i32* %storeaddr22, align 4 and   store i32 0, i32* %storeaddr22, align 4
; CHECK-NEXT: LV: Creating an interleave group with:  store i32 %tmp24, i32* %storeaddr22, align 4
; CHECK-NEXT: LV: Inserted:  store i32 %tmp14, i32* %storeaddr12, align 4
; CHECK-NEXT:     into the interleave group with  store i32 %tmp24, i32* %storeaddr22, align 4
; CHECK-NEXT: LV: Invalidated store group due to dependence between   store i32 %tmp14, i32* %storeaddr12, align 4 and   store i32 0, i32* %storeaddr12, align 4


define void @test(i8* nonnull align 8 dereferenceable_or_null(24) %arg) {
bb:
  %tmp = getelementptr inbounds i8, i8* %arg, i64 16
  %tmp1 = bitcast i8* %tmp to i8**
  %tmp2 = load i8*, i8** %tmp1, align 8
  %tmp3 = getelementptr inbounds i8, i8* %arg, i64 8
  %tmp4 = bitcast i8* %tmp3 to i8**
  %tmp5 = load i8*, i8** %tmp4, align 8
  %tmp6 = getelementptr inbounds i8, i8* %tmp5, i64 12
  %tmp7 = bitcast i8* %tmp6 to i32*
  %tmp8 = getelementptr inbounds i8, i8* %tmp2, i64 12
  br label %header

header:                                              ; preds = %latch, %bb
  %tmp10 = phi i64 [ %tmp41, %latch ], [ 3, %bb ]
  %tmp11 = add nsw i64 %tmp10, -1
  %storeaddr12 = getelementptr inbounds i32, i32* %tmp7, i64 %tmp11
  %tmp13 = load i32, i32* %storeaddr12, align 4
  %tmp14 = add i32 %tmp13, 1
  store i32 %tmp14, i32* %storeaddr12, align 4
  %tmp15 = icmp slt i32 %tmp13, 1
  %tmp16 = xor i1 %tmp15, true
  %tmp17 = zext i1 %tmp16 to i8
  %tmp18 = getelementptr inbounds i8, i8* %tmp8, i64 %tmp10
  store i8 %tmp17, i8* %tmp18, align 1
  br i1 %tmp15, label %bb19, label %bb20

bb19:                                             ; preds = %header
  store i32 0, i32* %storeaddr12, align 4
  br label %bb20

bb20:                                             ; preds = %bb19, %header
  %tmp21 = add nuw nsw i64 %tmp10, 1
  %storeaddr22 = getelementptr inbounds i32, i32* %tmp7, i64 %tmp10
  %tmp23 = load i32, i32* %storeaddr22, align 4
  %tmp24 = add i32 %tmp23, 1
  store i32 %tmp24, i32* %storeaddr22, align 4
  %tmp25 = icmp slt i32 %tmp23, 1
  %tmp26 = xor i1 %tmp25, true
  %tmp27 = zext i1 %tmp26 to i8
  %tmp28 = getelementptr inbounds i8, i8* %tmp8, i64 %tmp21
  store i8 %tmp27, i8* %tmp28, align 1
  br i1 %tmp25, label %bb29, label %bb30

bb29:                                             ; preds = %bb20
  store i32 0, i32* %storeaddr22, align 4
  br label %bb30

bb30:                                             ; preds = %bb29, %bb20
  %tmp31 = add nuw nsw i64 %tmp10, 2
  %storeaddr32 = getelementptr inbounds i32, i32* %tmp7, i64 %tmp21
  %tmp33 = load i32, i32* %storeaddr32, align 4
  %tmp34 = add i32 %tmp33, 1
  store i32 %tmp34, i32* %storeaddr32, align 4
  %tmp35 = icmp slt i32 %tmp33, 1
  %tmp36 = xor i1 %tmp35, true
  %tmp37 = zext i1 %tmp36 to i8
  %tmp38 = getelementptr inbounds i8, i8* %tmp8, i64 %tmp31
  store i8 %tmp37, i8* %tmp38, align 1
  br i1 %tmp35, label %bb39, label %latch

bb39:                                             ; preds = %bb30
  store i32 0, i32* %storeaddr32, align 4
  br label %latch

latch:                                             ; preds = %bb39, %bb30
  %tmp41 = add nuw nsw i64 %tmp10, 3
  %tmp42 = icmp ugt i64 %tmp31, 67
  br i1 %tmp42, label %exit, label %header

exit:                                             ; preds = %latch
  ret void
}
