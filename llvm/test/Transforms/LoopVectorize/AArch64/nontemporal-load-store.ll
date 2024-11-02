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

define i4 @test_i4_load(i4* %ddst) {
; CHECK-LABEL: define i4 @test_i4_load
; CHECK-NOT: vector.body:
; CHECk: ret i4 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i4 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i4, i4* %ddst, i64 %indvars.iv
  %l = load i4, i4* %arrayidx, align 1, !nontemporal !8
  %add = add i4 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i4 %add
}

define i8 @test_load_i8(i8* %ddst) {
; CHECK-LABEL: @test_load_i8(
; CHECK:   vector.body:
; CHECK: load <4 x i8>, <4 x i8>* {{.*}}, align 1, !nontemporal !0
; CHECk: ret i8 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i8 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i8, i8* %ddst, i64 %indvars.iv
  %l = load i8, i8* %arrayidx, align 1, !nontemporal !8
  %add = add i8 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i8 %add
}

define half @test_half_load(half* %ddst) {
; CHECK-LABEL: @test_half_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x half>, <4 x half>* {{.*}}, align 2, !nontemporal !0
; CHECk: ret half %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi half [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds half, half* %ddst, i64 %indvars.iv
  %l = load half, half* %arrayidx, align 2, !nontemporal !8
  %add = fadd half %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret half %add
}

define i16 @test_i16_load(i16* %ddst) {
; CHECK-LABEL: @test_i16_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x i16>, <4 x i16>* {{.*}}, align 2, !nontemporal !0
; CHECk: ret i16 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i16 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i16, i16* %ddst, i64 %indvars.iv
  %l = load i16, i16* %arrayidx, align 2, !nontemporal !8
  %add = add i16 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i16 %add
}

define i32 @test_i32_load(i32* %ddst) {
; CHECK-LABEL: @test_i32_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x i32>, <4 x i32>* {{.*}}, align 4, !nontemporal !0
; CHECk: ret i32 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %ddst, i64 %indvars.iv
  %l = load i32, i32* %arrayidx, align 4, !nontemporal !8
  %add = add i32 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add
}

define i33 @test_i33_load(i33* %ddst) {
; CHECK-LABEL: @test_i33_load
; CHECK-NOT:   vector.body:
; CHECk: ret i33 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i33 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i33, i33* %ddst, i64 %indvars.iv
  %l = load i33, i33* %arrayidx, align 4, !nontemporal !8
  %add = add i33 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i33 %add
}

define i40 @test_i40_load(i40* %ddst) {
; CHECK-LABEL: @test_i40_load
; CHECK-NOT:   vector.body:
; CHECk: ret i40 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i40 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i40, i40* %ddst, i64 %indvars.iv
  %l = load i40, i40* %arrayidx, align 4, !nontemporal !8
  %add = add i40 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i40 %add
}

define i64 @test_i64_load(i64* %ddst) {
; CHECK-LABEL: @test_i64_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x i64>, <4 x i64>* {{.*}}, align 4, !nontemporal !0
; CHECk: ret i64 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i64, i64* %ddst, i64 %indvars.iv
  %l = load i64, i64* %arrayidx, align 4, !nontemporal !8
  %add = add i64 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i64 %add
}

define double @test_double_load(double* %ddst) {
; CHECK-LABEL: @test_double_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x double>, <4 x double>* {{.*}}, align 4, !nontemporal !0
; CHECk: ret double %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi double [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, double* %ddst, i64 %indvars.iv
  %l = load double, double* %arrayidx, align 4, !nontemporal !8
  %add = fadd double %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add
}

define i128 @test_i128_load(i128* %ddst) {
; CHECK-LABEL: @test_i128_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x i128>, <4 x i128>* {{.*}}, align 4, !nontemporal !0
; CHECk: ret i128 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i128 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i128, i128* %ddst, i64 %indvars.iv
  %l = load i128, i128* %arrayidx, align 4, !nontemporal !8
  %add = add i128 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i128 %add
}

define i256 @test_256_load(i256* %ddst) {
; CHECK-LABEL: @test_256_load
; CHECK-NOT:   vector.body:
; CHECk: ret i256 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i256 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i256, i256* %ddst, i64 %indvars.iv
  %l = load i256, i256* %arrayidx, align 4, !nontemporal !8
  %add = add i256 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i256 %add
}

!8 = !{i32 1}
