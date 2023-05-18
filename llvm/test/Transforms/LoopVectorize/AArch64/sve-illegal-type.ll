; RUN: opt < %s -passes=loop-vectorize -mattr=+sve -force-vector-width=4 -pass-remarks-analysis=loop-vectorize \
; RUN:   -prefer-predicate-over-epilogue=scalar-epilogue -S 2>%t | FileCheck %s
; RUN: cat %t | FileCheck %s -check-prefix=CHECK-REMARKS
target triple = "aarch64-linux-gnu"

; CHECK-REMARKS: Scalable vectorization is not supported for all element types found in this loop
define dso_local void @loop_sve_i128(ptr nocapture %ptr, i64 %N) {
; CHECK-LABEL: @loop_sve_i128
; CHECK: vector.body
; CHECK:  %[[LOAD1:.*]] = load i128, ptr {{.*}}
; CHECK-NEXT: %[[LOAD2:.*]] = load i128, ptr {{.*}}
; CHECK-NEXT: %[[ADD1:.*]] = add nsw i128 %[[LOAD1]], 42
; CHECK-NEXT: %[[ADD2:.*]] = add nsw i128 %[[LOAD2]], 42
; CHECK-NEXT: store i128 %[[ADD1]], ptr {{.*}}
; CHECK-NEXT: store i128 %[[ADD2]], ptr {{.*}}
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i128, ptr %ptr, i64 %iv
  %0 = load i128, ptr %arrayidx, align 16
  %add = add nsw i128 %0, 42
  store i128 %add, ptr %arrayidx, align 16
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

; CHECK-REMARKS: Scalable vectorization is not supported for all element types found in this loop
define dso_local void @loop_sve_f128(ptr nocapture %ptr, i64 %N) {
; CHECK-LABEL: @loop_sve_f128
; CHECK: vector.body
; CHECK: %[[LOAD1:.*]] = load fp128, ptr
; CHECK-NEXT: %[[LOAD2:.*]] = load fp128, ptr
; CHECK-NEXT: %[[FSUB1:.*]] = fsub fp128 %[[LOAD1]], 0xL00000000000000008000000000000000
; CHECK-NEXT: %[[FSUB2:.*]] = fsub fp128 %[[LOAD2]], 0xL00000000000000008000000000000000
; CHECK-NEXT: store fp128 %[[FSUB1]], ptr {{.*}}
; CHECK-NEXT: store fp128 %[[FSUB2]], ptr {{.*}}
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds fp128, ptr %ptr, i64 %iv
  %0 = load fp128, ptr %arrayidx, align 16
  %add = fsub fp128 %0, 0xL00000000000000008000000000000000
  store fp128 %add, ptr %arrayidx, align 16
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

; CHECK-REMARKS: Scalable vectorization is not supported for all element types found in this loop
define dso_local void @loop_invariant_sve_i128(ptr nocapture %ptr, i128 %val, i64 %N) {
; CHECK-LABEL: @loop_invariant_sve_i128
; CHECK: vector.body
; CHECK: %[[GEP1:.*]] = getelementptr inbounds i128, ptr %ptr
; CHECK-NEXT: %[[GEP2:.*]] = getelementptr inbounds i128, ptr %ptr
; CHECK-NEXT: store i128 %val, ptr %[[GEP1]]
; CHECK-NEXT: store i128 %val, ptr %[[GEP2]]
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i128, ptr %ptr, i64 %iv
  store i128 %val, ptr %arrayidx, align 16
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

; CHECK-REMARKS: Scalable vectorization is not supported for all element types found in this loop
define void @uniform_store_i1(ptr noalias %dst, ptr noalias %start, i64 %N) {
; CHECK-LABEL: @uniform_store_i1
; CHECK: vector.body
; CHECK: %[[GEP:.*]] = getelementptr inbounds i64, <64 x ptr> {{.*}}, i64 1
; CHECK: %[[ICMP:.*]] = icmp eq <64 x ptr> %[[GEP]], %[[SPLAT:.*]]
; CHECK: %[[EXTRACT1:.*]] = extractelement <64 x i1> %[[ICMP]], i32 63
; CHECK: store i1 %[[EXTRACT1]], ptr %dst
; CHECK-NOT: vscale
entry:
  br label %for.body

for.body:
  %first.sroa = phi ptr [ %incdec.ptr, %for.body ], [ %start, %entry ]
  %iv = phi i64 [ %iv.next, %for.body ], [ 0, %entry ]
  %iv.next = add i64 %iv, 1
  %0 = load i64, ptr %first.sroa
  %incdec.ptr = getelementptr inbounds i64, ptr %first.sroa, i64 1
  %cmp.not = icmp eq ptr %incdec.ptr, %start
  store i1 %cmp.not, ptr %dst
  %cmp = icmp ult i64 %iv, %N
  br i1 %cmp, label %for.body, label %end, !llvm.loop !0

end:
  ret void
}

define dso_local void @loop_fixed_width_i128(ptr nocapture %ptr, i64 %N) {
; CHECK-LABEL: @loop_fixed_width_i128
; CHECK: load <4 x i128>, ptr
; CHECK: add nsw <4 x i128> {{.*}}, <i128 42, i128 42, i128 42, i128 42>
; CHECK: store <4 x i128> {{.*}} ptr
; CHECK-NOT: vscale
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i128, ptr %ptr, i64 %iv
  %0 = load i128, ptr %arrayidx, align 16
  %add = add nsw i128 %0, 42
  store i128 %add, ptr %arrayidx, align 16
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
