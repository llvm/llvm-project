; RUN: opt %loadNPMPolly -aa-pipeline=basic-aa -passes=polly-codegen -S < %s | FileCheck %s
;
; Verify that we remove the misc intrinsics  from the optimized SCoP.
;
; CHECK: for.body:
; CHECK:   call ptr @llvm.invariant.start
; CHECK: for.body4:
; CHECK:   call void @llvm.assume
; CHECK:   call i1 @llvm.expect.i1
; CHECK:   call void @llvm.donothing
; CHECK: for.end:
; CHECK:   call void @llvm.invariant.end
; CHECK-NOT: call void @llvm.{{.*}}
;
;    int A[1024];
;    void jd() {
;      for (int i = 0; i < 1024; i++) {
;        int tmp[1024];
;        for (int j = i; j < 1024; j++)
;          tmp[i] += A[j];
;        A[i] = tmp[i];
;      }
;    }
;
; ModuleID = 'test/Isl/CodeGen/lifetime_intrinsics.ll'
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

@A = common global [1024 x i32] zeroinitializer, align 16

; Function Attrs: nounwind uwtable
define void @jd() #0 {
entry:
  %tmp = alloca [1024 x i32], align 16
  br label %for.cond

for.cond:                                         ; preds = %for.inc11, %entry
  %indvars.iv3 = phi i64 [ %indvars.iv.next4, %for.inc11 ], [ 0, %entry ]
  %exitcond5 = icmp ne i64 %indvars.iv3, 1024
  br i1 %exitcond5, label %for.body, label %for.end13

for.body:                                         ; preds = %for.cond
  %lis = call ptr @llvm.invariant.start(i64 4096, ptr @A) #1
  br label %for.cond2

for.cond2:                                        ; preds = %for.inc, %for.body
  %indvars.iv1 = phi i64 [ %indvars.iv.next2, %for.inc ], [ %indvars.iv3, %for.body ]
  %lftr.wideiv = trunc i64 %indvars.iv1 to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 1024
  br i1 %exitcond, label %for.body4, label %for.end

for.body4:                                        ; preds = %for.cond2
  call void @llvm.assume(i1 %exitcond)
  call i1 @llvm.expect.i1(i1 %exitcond, i1 1)
  %arrayidx = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv1
  %tmp6 = load i32, ptr %arrayidx, align 4
  %arrayidx6 = getelementptr inbounds [1024 x i32], ptr %tmp, i64 0, i64 %indvars.iv3
  call void @llvm.donothing()
  %tmp7 = load i32, ptr %arrayidx6, align 4
  %add = add nsw i32 %tmp7, %tmp6
  store i32 %add, ptr %arrayidx6, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body4
  %indvars.iv.next2 = add nuw nsw i64 %indvars.iv1, 1
  br label %for.cond2

for.end:                                          ; preds = %for.cond2
  %arrayidx8 = getelementptr inbounds [1024 x i32], ptr %tmp, i64 0, i64 %indvars.iv3
  %tmp8 = load i32, ptr %arrayidx8, align 4
  %arrayidx10 = getelementptr inbounds [1024 x i32], ptr @A, i64 0, i64 %indvars.iv3
  call void @llvm.invariant.end(ptr %lis, i64 4096, ptr @A) #1
  store i32 %tmp8, ptr %arrayidx10, align 4
  br label %for.inc11

for.inc11:                                        ; preds = %for.end
  %indvars.iv.next4 = add nuw nsw i64 %indvars.iv3, 1
  br label %for.cond

for.end13:                                        ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind
declare void @llvm.donothing() #1

; Function Attrs: nounwind
declare void @llvm.assume(i1) #1

; Function Attrs: nounwind
declare i1 @llvm.expect.i1(i1, i1) #1

; Function Attrs: nounwind
declare ptr @llvm.invariant.start(i64, ptr nocapture) #1

; Function Attrs: nounwind
declare void @llvm.invariant.end(ptr, i64, ptr nocapture) #1

attributes #0 = { nounwind uwtable "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="true" "no-nans-fp-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="true" "use-soft-float"="false" }
attributes #1 = { nounwind }
