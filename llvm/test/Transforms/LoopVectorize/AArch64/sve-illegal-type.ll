; RUN: opt < %s -loop-vectorize -scalable-vectorization=on -mattr=+sve -force-vector-width=4 -pass-remarks-analysis=loop-vectorize -S 2>%t | FileCheck %s
; RUN: cat %t | FileCheck %s -check-prefix=CHECK-REMARKS
target triple = "aarch64-linux-gnu"

; CHECK-REMARKS: Scalable vectorization is not supported for all element types found in this loop
define dso_local void @loop_sve_i128(i128* nocapture %ptr, i64 %N) {
; CHECK-LABEL: @loop_sve_i128
; CHECK: vector.body
; CHECK:  %[[LOAD1:.*]] = load i128, i128* {{.*}}
; CHECK-NEXT: %[[LOAD2:.*]] = load i128, i128* {{.*}}
; CHECK-NEXT: %[[ADD1:.*]] = add nsw i128 %[[LOAD1]], 42
; CHECK-NEXT: %[[ADD2:.*]] = add nsw i128 %[[LOAD2]], 42
; CHECK-NEXT: store i128 %[[ADD1]], i128* {{.*}}
; CHECK-NEXT: store i128 %[[ADD2]], i128* {{.*}}
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i128, i128* %ptr, i64 %iv
  %0 = load i128, i128* %arrayidx, align 16
  %add = add nsw i128 %0, 42
  store i128 %add, i128* %arrayidx, align 16
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

; CHECK-REMARKS: Scalable vectorization is not supported for all element types found in this loop
define dso_local void @loop_sve_f128(fp128* nocapture %ptr, i64 %N) {
; CHECK-LABEL: @loop_sve_f128
; CHECK: vector.body
; CHECK: %[[LOAD1:.*]] = load fp128, fp128*
; CHECK-NEXT: %[[LOAD2:.*]] = load fp128, fp128*
; CHECK-NEXT: %[[FSUB1:.*]] = fsub fp128 %[[LOAD1]], 0xL00000000000000008000000000000000
; CHECK-NEXT: %[[FSUB2:.*]] = fsub fp128 %[[LOAD2]], 0xL00000000000000008000000000000000
; CHECK-NEXT: store fp128 %[[FSUB1]], fp128* {{.*}}
; CHECK-NEXT: store fp128 %[[FSUB2]], fp128* {{.*}}
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds fp128, fp128* %ptr, i64 %iv
  %0 = load fp128, fp128* %arrayidx, align 16
  %add = fsub fp128 %0, 0xL00000000000000008000000000000000
  store fp128 %add, fp128* %arrayidx, align 16
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

; CHECK-REMARKS: Scalable vectorization is not supported for all element types found in this loop
define dso_local void @loop_invariant_sve_i128(i128* nocapture %ptr, i128 %val, i64 %N) {
; CHECK-LABEL: @loop_invariant_sve_i128
; CHECK: vector.body
; CHECK: %[[GEP1:.*]] = getelementptr inbounds i128, i128* %ptr
; CHECK-NEXT: %[[GEP2:.*]] = getelementptr inbounds i128, i128* %ptr
; CHECK-NEXT: store i128 %val, i128* %[[GEP1]]
; CHECK-NEXT: store i128 %val, i128* %[[GEP2]]
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i128, i128* %ptr, i64 %iv
  store i128 %val, i128* %arrayidx, align 16
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body, !llvm.loop !0

for.end:
  ret void
}

define dso_local void @loop_fixed_width_i128(i128* nocapture %ptr, i64 %N) {
; CHECK-LABEL: @loop_fixed_width_i128
; CHECK: load <4 x i128>, <4 x i128>*
; CHECK: add nsw <4 x i128> {{.*}}, <i128 42, i128 42, i128 42, i128 42>
; CHECK: store <4 x i128> {{.*}} <4 x i128>*
; CHECK-NOT: vscale
entry:
  br label %for.body

for.body:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i128, i128* %ptr, i64 %iv
  %0 = load i128, i128* %arrayidx, align 16
  %add = add nsw i128 %0, 42
  store i128 %add, i128* %arrayidx, align 16
  %iv.next = add i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %for.end, label %for.body

for.end:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.vectorize.scalable.enable", i1 true}
