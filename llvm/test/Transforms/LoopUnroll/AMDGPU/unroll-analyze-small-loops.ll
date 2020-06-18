; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -loop-unroll -unroll-threshold=150 < %s | FileCheck %s

; Test that max iterations count to analyze (specific for the target)
; is enough to make the inner loop completely unrolled
define hidden void @foo(float addrspace(1)* %ptrG, float addrspace(3)* %ptrL, i32 %A, i32 %A2, i32 %M) {
bb:
  br label %bb2

bb2:                                              ; preds = %bb7, %bb
  %i = phi i32 [ 0, %bb ], [ %i8, %bb7 ]
  br label %bb4

bb3:                                              ; preds = %bb7
  ret void

bb4:                                              ; preds = %bb10, %bb2
  %i5 = phi i32 [ 0, %bb2 ], [ %i11, %bb10 ]
  %i6 = add nuw nsw i32 %i5, %i
  br label %for.body

bb7:                                              ; preds = %bb10
  %i8 = add nuw nsw i32 %i, 1
  %i9 = icmp eq i32 %i8, 8
  br i1 %i9, label %bb3, label %bb2

bb10:                                             ; preds = %for.body
  %i11 = add nuw nsw i32 %i5, 1
  %cmpj = icmp ult i32 %i11, 8
  br i1 %cmpj, label %bb7, label %bb4

; CHECK: for.body:
; CHECK-NOT: %phi = phi {{.*}}
for.body:                                         ; preds = %for.body, %bb4
  %phi = phi i32 [ 0, %bb4 ], [ %k, %for.body ]
  %mul = shl nuw nsw i32 %phi, 5
  %add1 = add i32 %A, %mul
  %add2 = add i32 %add1, %M
  %arrayidx = getelementptr inbounds float, float addrspace(3)* %ptrL, i32 %add2
  %bc = bitcast float addrspace(3)* %arrayidx to i32 addrspace(3)*
  %ld = load i32, i32 addrspace(3)* %bc, align 4
  %mul2 = shl nuw nsw i32 %phi, 3
  %add3 = add nuw nsw i32 %mul2, %A2
  %arrayidx2 = getelementptr inbounds float, float addrspace(1)* %ptrG, i32 %add3
  %bc2 = bitcast float addrspace(1)* %arrayidx2 to i32 addrspace(1)*
  store i32 %ld, i32 addrspace(1)* %bc2, align 4
  %k = add nuw nsw i32 %phi, 1
  %cmpk = icmp ult i32 %k, 32
  br i1 %cmpk, label %for.body, label %bb10
}
