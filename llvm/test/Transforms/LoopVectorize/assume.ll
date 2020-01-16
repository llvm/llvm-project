; RUN: opt < %s  -loop-vectorize -force-vector-width=4 -force-vector-interleave=2  -S | FileCheck %s

define void @test1(float* noalias nocapture %a, float* noalias nocapture readonly %b) {
entry:
  br label %for.body

; CHECK-LABEL: @test1
; CHECK: vector.body:
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: for.body:
; CHECK: ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, float* %b, i64 %indvars.iv
  %0 = load float, float* %arrayidx, align 4
  %cmp1 = fcmp ogt float %0, 1.000000e+02
  tail call void @llvm.assume(i1 %cmp1)
  %add = fadd float %0, 1.000000e+00
  %arrayidx5 = getelementptr inbounds float, float* %a, i64 %indvars.iv
  store float %add, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

declare void @llvm.assume(i1) #0

attributes #0 = { nounwind willreturn }

%struct.data = type { float*, float* }

define void @test2(%struct.data* nocapture readonly %d) {
entry:
  %b = getelementptr inbounds %struct.data, %struct.data* %d, i64 0, i32 1
  %0 = load float*, float** %b, align 8
  %ptrint = ptrtoint float* %0 to i64
  %maskedptr = and i64 %ptrint, 31
  %maskcond = icmp eq i64 %maskedptr, 0
  %a = getelementptr inbounds %struct.data, %struct.data* %d, i64 0, i32 0
  %1 = load float*, float** %a, align 8
  %ptrint2 = ptrtoint float* %1 to i64
  %maskedptr3 = and i64 %ptrint2, 31
  %maskcond4 = icmp eq i64 %maskedptr3, 0
  br label %for.body

; CHECK-LABEL: @test2
; CHECK: vector.body:
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: @llvm.assume
; CHECK: for.body:
; CHECK: ret void

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  tail call void @llvm.assume(i1 %maskcond)
  %arrayidx = getelementptr inbounds float, float* %0, i64 %indvars.iv
  %2 = load float, float* %arrayidx, align 4
  %add = fadd float %2, 1.000000e+00
  tail call void @llvm.assume(i1 %maskcond4)
  %arrayidx5 = getelementptr inbounds float, float* %1, i64 %indvars.iv
  store float %add, float* %arrayidx5, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv, 1599
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}
