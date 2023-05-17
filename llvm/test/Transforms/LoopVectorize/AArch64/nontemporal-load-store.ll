; RUN: opt -passes=loop-vectorize -mtriple=arm64-apple-iphones -force-vector-width=4 -force-vector-interleave=1 %s -S | FileCheck %s

; Vectors with i4 elements may not legal with nontemporal stores.
define void @test_i4_store(ptr %ddst) {
; CHECK-LABEL: define void @test_i4_store(
; CHECK-NOT:   vector.body:
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i4, ptr %ddst.addr, i64 1
  store i4 10, ptr %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i8_store(ptr %ddst) {
; CHECK-LABEL: define void @test_i8_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x i8> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i8, ptr %ddst.addr, i64 1
  store i8 10, ptr %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_half_store(ptr %ddst) {
; CHECK-LABEL: define void @test_half_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x half> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds half, ptr %ddst.addr, i64 1
  store half 10.0, ptr %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i16_store(ptr %ddst) {
; CHECK-LABEL: define void @test_i16_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x i16> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i16, ptr %ddst.addr, i64 1
  store i16 10, ptr %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i32_store(ptr nocapture %ddst) {
; CHECK-LABEL: define void @test_i32_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <16 x i32> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr3, %for.body ]
  %incdec.ptr = getelementptr inbounds i32, ptr %ddst.addr, i64 1
  store i32 10, ptr %ddst.addr, align 4, !nontemporal !8
  %incdec.ptr1 = getelementptr inbounds i32, ptr %ddst.addr, i64 2
  store i32 20, ptr %incdec.ptr, align 4, !nontemporal !8
  %incdec.ptr2 = getelementptr inbounds i32, ptr %ddst.addr, i64 3
  store i32 30, ptr %incdec.ptr1, align 4, !nontemporal !8
  %incdec.ptr3 = getelementptr inbounds i32, ptr %ddst.addr, i64 4
  store i32 40, ptr %incdec.ptr2, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i33_store(ptr nocapture %ddst) {
; CHECK-LABEL: define void @test_i33_store(
; CHECK-NOT:   vector.body:
; CHECK:         ret
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr3, %for.body ]
  %incdec.ptr = getelementptr inbounds i33, ptr %ddst.addr, i64 1
  store i33 10, ptr %ddst.addr, align 4, !nontemporal !8
  %incdec.ptr1 = getelementptr inbounds i33, ptr %ddst.addr, i64 2
  store i33 20, ptr %incdec.ptr, align 4, !nontemporal !8
  %incdec.ptr2 = getelementptr inbounds i33, ptr %ddst.addr, i64 3
  store i33 30, ptr %incdec.ptr1, align 4, !nontemporal !8
  %incdec.ptr3 = getelementptr inbounds i33, ptr %ddst.addr, i64 4
  store i33 40, ptr %incdec.ptr2, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 3
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i40_store(ptr nocapture %ddst) {
; CHECK-LABEL: define void @test_i40_store(
; CHECK-NOT:   vector.body:
; CHECK:         ret
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr3, %for.body ]
  %incdec.ptr = getelementptr inbounds i40, ptr %ddst.addr, i64 1
  store i40 10, ptr %ddst.addr, align 4, !nontemporal !8
  %incdec.ptr1 = getelementptr inbounds i40, ptr %ddst.addr, i64 2
  store i40 20, ptr %incdec.ptr, align 4, !nontemporal !8
  %incdec.ptr2 = getelementptr inbounds i40, ptr %ddst.addr, i64 3
  store i40 30, ptr %incdec.ptr1, align 4, !nontemporal !8
  %incdec.ptr3 = getelementptr inbounds i40, ptr %ddst.addr, i64 4
  store i40 40, ptr %incdec.ptr2, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 3
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}
define void @test_i64_store(ptr nocapture %ddst) local_unnamed_addr #0 {
; CHECK-LABEL: define void @test_i64_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x i64> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i64, ptr %ddst.addr, i64 1
  store i64 10, ptr %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_double_store(ptr %ddst) {
; CHECK-LABEL: define void @test_double_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x double> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds double, ptr %ddst.addr, i64 1
  store double 10.0, ptr %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i128_store(ptr %ddst) {
; CHECK-LABEL: define void @test_i128_store(
; CHECK-LABEL: vector.body:
; CHECK:         store <4 x i128> {{.*}} !nontemporal !0
; CHECK:         br
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i128, ptr %ddst.addr, i64 1
  store i128 10, ptr %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define void @test_i256_store(ptr %ddst) {
; CHECK-LABEL: define void @test_i256_store(
; CHECK-NOT:   vector.body:
; CHECK:        ret void
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %i = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %ddst.addr = phi ptr [ %ddst, %entry ], [ %incdec.ptr, %for.body ]
  %incdec.ptr = getelementptr inbounds i256, ptr %ddst.addr, i64 1
  store i256 10, ptr %ddst.addr, align 4, !nontemporal !8
  %add = add nuw nsw i32 %i, 4
  %cmp = icmp ult i32 %i, 4092
  br i1 %cmp, label %for.body, label %for.cond.cleanup

for.cond.cleanup:                                 ; preds = %for.body
  ret void
}

define i4 @test_i4_load(ptr %ddst) {
; CHECK-LABEL: define i4 @test_i4_load
; CHECK-NOT: vector.body:
; CHECk: ret i4 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i4 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i4, ptr %ddst, i64 %indvars.iv
  %l = load i4, ptr %arrayidx, align 1, !nontemporal !8
  %add = add i4 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i4 %add
}

define i8 @test_load_i8(ptr %ddst) {
; CHECK-LABEL: @test_load_i8(
; CHECK:   vector.body:
; CHECK: load <4 x i8>, ptr {{.*}}, align 1, !nontemporal !0
; CHECk: ret i8 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i8 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i8, ptr %ddst, i64 %indvars.iv
  %l = load i8, ptr %arrayidx, align 1, !nontemporal !8
  %add = add i8 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i8 %add
}

define half @test_half_load(ptr %ddst) {
; CHECK-LABEL: @test_half_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x half>, ptr {{.*}}, align 2, !nontemporal !0
; CHECk: ret half %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi half [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds half, ptr %ddst, i64 %indvars.iv
  %l = load half, ptr %arrayidx, align 2, !nontemporal !8
  %add = fadd half %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret half %add
}

define i16 @test_i16_load(ptr %ddst) {
; CHECK-LABEL: @test_i16_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x i16>, ptr {{.*}}, align 2, !nontemporal !0
; CHECk: ret i16 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i16 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i16, ptr %ddst, i64 %indvars.iv
  %l = load i16, ptr %arrayidx, align 2, !nontemporal !8
  %add = add i16 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i16 %add
}

define i32 @test_i32_load(ptr %ddst) {
; CHECK-LABEL: @test_i32_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x i32>, ptr {{.*}}, align 4, !nontemporal !0
; CHECk: ret i32 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i32 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i32, ptr %ddst, i64 %indvars.iv
  %l = load i32, ptr %arrayidx, align 4, !nontemporal !8
  %add = add i32 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i32 %add
}

define i33 @test_i33_load(ptr %ddst) {
; CHECK-LABEL: @test_i33_load
; CHECK-NOT:   vector.body:
; CHECk: ret i33 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i33 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i33, ptr %ddst, i64 %indvars.iv
  %l = load i33, ptr %arrayidx, align 4, !nontemporal !8
  %add = add i33 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i33 %add
}

define i40 @test_i40_load(ptr %ddst) {
; CHECK-LABEL: @test_i40_load
; CHECK-NOT:   vector.body:
; CHECk: ret i40 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i40 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i40, ptr %ddst, i64 %indvars.iv
  %l = load i40, ptr %arrayidx, align 4, !nontemporal !8
  %add = add i40 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i40 %add
}

define i64 @test_i64_load(ptr %ddst) {
; CHECK-LABEL: @test_i64_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x i64>, ptr {{.*}}, align 4, !nontemporal !0
; CHECk: ret i64 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i64 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i64, ptr %ddst, i64 %indvars.iv
  %l = load i64, ptr %arrayidx, align 4, !nontemporal !8
  %add = add i64 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i64 %add
}

define double @test_double_load(ptr %ddst) {
; CHECK-LABEL: @test_double_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x double>, ptr {{.*}}, align 4, !nontemporal !0
; CHECk: ret double %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi double [ 0.0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds double, ptr %ddst, i64 %indvars.iv
  %l = load double, ptr %arrayidx, align 4, !nontemporal !8
  %add = fadd double %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret double %add
}

define i128 @test_i128_load(ptr %ddst) {
; CHECK-LABEL: @test_i128_load
; CHECK-LABEL:   vector.body:
; CHECK: load <4 x i128>, ptr {{.*}}, align 4, !nontemporal !0
; CHECk: ret i128 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i128 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i128, ptr %ddst, i64 %indvars.iv
  %l = load i128, ptr %arrayidx, align 4, !nontemporal !8
  %add = add i128 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i128 %add
}

define i256 @test_256_load(ptr %ddst) {
; CHECK-LABEL: @test_256_load
; CHECK-NOT:   vector.body:
; CHECk: ret i256 %{{.*}}
;
entry:
  br label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %acc.08 = phi i256 [ 0, %entry ], [ %add, %for.body ]
  %arrayidx = getelementptr inbounds i256, ptr %ddst, i64 %indvars.iv
  %l = load i256, ptr %arrayidx, align 4, !nontemporal !8
  %add = add i256 %l, %acc.08
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, 4092
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body

for.cond.cleanup:                                 ; preds = %for.body
  ret i256 %add
}

!8 = !{i32 1}
