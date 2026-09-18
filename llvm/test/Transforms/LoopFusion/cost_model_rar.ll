; RUN: opt -passes=loop-simplify,loop-fusion -loop-fusion-cost-model -disable-output -pass-remarks=loop-fusion -pass-remarks-missed=loop-fusion < %s 2>&1 | FileCheck %s

; CHECK: [exact_affine_rar]{{.*}}Loops fused
; CHECK: [shifted_start]{{.*}}found 0 cross-loop reused values
; CHECK: [different_stride]{{.*}}found 0 cross-loop reused values
; CHECK: [different_type]{{.*}}found 0 cross-loop reused values
; CHECK: [non_affine]{{.*}}found 0 cross-loop reused values

; Equal load types, starts, and strides form an exact affine RAR match.
define void @exact_affine_rar(ptr noalias %src, ptr noalias %dst1,
                              ptr noalias %dst2, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1 ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %value1, ptr %dst.gep1, align 4
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2 ]
  %src.gep2 = getelementptr inbounds i32, ptr %src, i64 %i2
  %value2 = load i32, ptr %src.gep2, align 4
  %dst.gep2 = getelementptr inbounds i32, ptr %dst2, i64 %i2
  store i32 %value2, ptr %dst.gep2, align 4
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

; A one-element offset changes the SCEV start and is not exact RAR.
define void @shifted_start(ptr noalias %src, ptr noalias %dst1,
                           ptr noalias %dst2, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1 ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %value1, ptr %dst.gep1, align 4
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2 ]
  %shifted = add nuw nsw i64 %i2, 1
  %src.gep2 = getelementptr inbounds i32, ptr %src, i64 %shifted
  %value2 = load i32, ptr %src.gep2, align 4
  %dst.gep2 = getelementptr inbounds i32, ptr %dst2, i64 %i2
  store i32 %value2, ptr %dst.gep2, align 4
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

; A different per-iteration stride is not exact RAR.
define void @different_stride(ptr noalias %src, ptr noalias %dst1,
                              ptr noalias %dst2, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1 ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %value1, ptr %dst.gep1, align 4
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2 ]
  %strided = shl nuw nsw i64 %i2, 1
  %src.gep2 = getelementptr inbounds i32, ptr %src, i64 %strided
  %value2 = load i32, ptr %src.gep2, align 4
  %dst.gep2 = getelementptr inbounds i32, ptr %dst2, i64 %i2
  store i32 %value2, ptr %dst.gep2, align 4
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

; Equal byte addresses with different loaded types do not represent reuse of
; the same value.
define void @different_type(ptr noalias %src, ptr noalias %dst1,
                            ptr noalias %dst2, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1 ]
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %i1
  %value1 = load i32, ptr %src.gep1, align 4
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %value1, ptr %dst.gep1, align 4
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2 ]
  %src.gep2 = getelementptr inbounds float, ptr %src, i64 %i2
  %value2 = load float, ptr %src.gep2, align 4
  %dst.gep2 = getelementptr inbounds float, ptr %dst2, i64 %i2
  store float %value2, ptr %dst.gep2, align 4
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}

; Equal quadratic accesses are outside the affine RAR model.
define void @non_affine(ptr noalias %src, ptr noalias %dst1,
                        ptr noalias %dst2, i64 %n) {
entry:
  br label %loop1

loop1:
  %i1 = phi i64 [ 0, %entry ], [ %i1.next, %loop1 ]
  %square1 = mul nuw nsw i64 %i1, %i1
  %src.gep1 = getelementptr inbounds i32, ptr %src, i64 %square1
  %value1 = load i32, ptr %src.gep1, align 4
  %dst.gep1 = getelementptr inbounds i32, ptr %dst1, i64 %i1
  store i32 %value1, ptr %dst.gep1, align 4
  %i1.next = add nuw nsw i64 %i1, 1
  %cmp1 = icmp ult i64 %i1.next, %n
  br i1 %cmp1, label %loop1, label %loop2.preheader

loop2.preheader:
  br label %loop2

loop2:
  %i2 = phi i64 [ 0, %loop2.preheader ], [ %i2.next, %loop2 ]
  %square2 = mul nuw nsw i64 %i2, %i2
  %src.gep2 = getelementptr inbounds i32, ptr %src, i64 %square2
  %value2 = load i32, ptr %src.gep2, align 4
  %dst.gep2 = getelementptr inbounds i32, ptr %dst2, i64 %i2
  store i32 %value2, ptr %dst.gep2, align 4
  %i2.next = add nuw nsw i64 %i2, 1
  %cmp2 = icmp ult i64 %i2.next, %n
  br i1 %cmp2, label %loop2, label %exit

exit:
  ret void
}
