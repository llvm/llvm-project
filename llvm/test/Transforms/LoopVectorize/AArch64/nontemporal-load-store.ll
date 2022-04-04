; RUN: opt -loop-vectorize -mtriple=arm64-apple-iphones -force-vector-width=4 -force-vector-interleave=1 %s -S | FileCheck %s

; Vectors with i4 elements may not legal with nontemporal stores.
define void @test_i4_store(i4* %ddst) {
; CHECK-LABEL: define void @test_i4_store(
; CHECK-NOT:   vector.body:
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i4* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i4, i4* %ddst.addr, i64 1
  store i4 10, i4* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i8_store(i8* %ddst) {
; CHECK-LABEL: define void @test_i8_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x i8> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i8* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, i8* %ddst.addr, i64 1
  store i8 10, i8* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_half_store(half* %ddst) {
; CHECK-LABEL: define void @test_half_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x half> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi half* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds half, half* %ddst.addr, i64 1
  store half 10.0, half* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i16_store(i16* %ddst) {
; CHECK-LABEL: define void @test_i16_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x i16> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i16* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i16, i16* %ddst.addr, i64 1
  store i16 10, i16* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i32_store(i32* nocapture %ddst) {
; CHECK-LABEL: define void @test_i32_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <16 x i32> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i32* [ %ddst, %entry ], [ %incdec.ptr3, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, i32* %ddst.addr, i64 1
  store i32 10, i32* %ddst.addr, align 4, !nontemporal !8
  %incdec.ptr1 = getelementptr inbounds i32, i32* %ddst.addr, i64 2
  store i32 20, i32* %incdec.ptr, align 4, !nontemporal !8
  %incdec.ptr2 = getelementptr inbounds i32, i32* %ddst.addr, i64 3
  store i32 30, i32* %incdec.ptr1, align 4, !nontemporal !8
  %incdec.ptr3 = getelementptr inbounds i32, i32* %ddst.addr, i64 4
  store i32 40, i32* %incdec.ptr2, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i33_store(i33* nocapture %ddst) {
; CHECK-LABEL: define void @test_i33_store(
; CHECK-NOT:   vector.body:
; CHECK:         ret
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i33* [ %ddst, %entry ], [ %incdec.ptr3, %for.body ]
  %incdec.ptr = getelementptr inbounds i33, i33* %ddst.addr, i64 1
  store i33 10, i33* %ddst.addr, align 4, !nontemporal !8
  %incdec.ptr1 = getelementptr inbounds i33, i33* %ddst.addr, i64 2
  store i33 20, i33* %incdec.ptr, align 4, !nontemporal !8
  %incdec.ptr2 = getelementptr inbounds i33, i33* %ddst.addr, i64 3
  store i33 30, i33* %incdec.ptr1, align 4, !nontemporal !8
  %incdec.ptr3 = getelementptr inbounds i33, i33* %ddst.addr, i64 4
  store i33 40, i33* %incdec.ptr2, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 3
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i40_store(i40* nocapture %ddst) {
; CHECK-LABEL: define void @test_i40_store(
; CHECK-NOT:   vector.body:
; CHECK:         ret
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i40* [ %ddst, %entry ], [ %incdec.ptr3, %for.body ]
  %incdec.ptr = getelementptr inbounds i40, i40* %ddst.addr, i64 1
  store i40 10, i40* %ddst.addr, align 4, !nontemporal !8
  %incdec.ptr1 = getelementptr inbounds i40, i40* %ddst.addr, i64 2
  store i40 20, i40* %incdec.ptr, align 4, !nontemporal !8
  %incdec.ptr2 = getelementptr inbounds i40, i40* %ddst.addr, i64 3
  store i40 30, i40* %incdec.ptr1, align 4, !nontemporal !8
  %incdec.ptr3 = getelementptr inbounds i40, i40* %ddst.addr, i64 4
  store i40 40, i40* %incdec.ptr2, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 3
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}
define void @test_i64_store(i64* nocapture %ddst) local_unnamed_addr #0 {
; CHECK-LABEL: define void @test_i64_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x i64> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i64* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i64, i64* %ddst.addr, i64 1
  store i64 10, i64* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_double_store(double* %ddst) {
; CHECK-LABEL: define void @test_double_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x double> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi double* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds double, double* %ddst.addr, i64 1
  store double 10.0, double* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i128_store(i128* %ddst) {
; CHECK-LABEL: define void @test_i128_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x i128> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i128* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i128, i128* %ddst.addr, i64 1
  store i128 10, i128* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i256_store(i256* %ddst) {
; CHECK-LABEL: define void @test_i256_store(
; CHECK-NOT:   vector.body:
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi i256* [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i256, i256* %ddst.addr, i64 1
  store i256 10, i256* %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

!8 = !{i32 1}
