; RUN: opt -S -mtriple=amdgcn-unknown-amdhsa -passes=loop-unroll < %s | FileCheck %s

; Test that max iterations count to analyze (specific for the target)
; is enough to make the inner loop completely unrolled
; CHECK-LABEL: foo
define void @foo(ptr addrspace(5) %ptrB, ptr addrspace(5) %ptrC, i32 %A, i32 %A2, float %M) {
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

; CHECK-LABEL: for.body
; CHECK-NOT: %phi = phi {{.*}}
for.body:                                       ; preds = %bb4, %for.body
  %phi = phi i32 [ 0, %bb4 ], [ %inc, %for.body ]
  %mul = shl nuw nsw i32 %phi, 6
  %add = add i32 %A, %mul
  %arrayidx = getelementptr inbounds float, ptr addrspace(5) %ptrC, i32 %add
  %ld1 = load float, ptr addrspace(5) %arrayidx, align 4
  %mul2 = shl nuw nsw i32 %phi, 3
  %add2 = add i32 %A2, %mul2
  %arrayidx2 = getelementptr inbounds float, ptr addrspace(5) %ptrB, i32 %add2
  %ld2 = load float, ptr addrspace(5) %arrayidx2, align 4
  %mul3 = fmul contract float %M, %ld2
  %add3 = fadd contract float %ld1, %mul3
  store float %add3, ptr addrspace(5) %arrayidx, align 4
  %add1 = add nuw nsw i32 %add, 2048
  %arrayidx3 = getelementptr inbounds float, ptr addrspace(5) %ptrC, i32 %add1
  %ld3 = load float, ptr addrspace(5) %arrayidx3, align 4
  %mul4 = fmul contract float %ld2, %M
  %add4 = fadd contract float %ld3, %mul4
  store float %add4, ptr addrspace(5) %arrayidx3, align 4
  %inc = add nuw nsw i32 %phi, 1
  %cmpi = icmp ult i32 %phi, 31
  br i1 %cmpi, label %for.body, label %bb10
}
