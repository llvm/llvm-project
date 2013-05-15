; RUN: opt %loadPolly -basicaa -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed+gpu -enable-polly-gpgpu -polly-gpgpu-triple=nvptx64-unknown-unknown -polly-codegen < %s -S | FileCheck %s

;int A[128][128];
;
;int gpu_pure() {
;  int i,j;
;
;  for(i = 0; i < 128; i++)
;    for(j = 0; j < 128; j++)
;      A[i][j] = i*128 + j;
;
;  return 0;
;}
;
;int main() {
;  int b = gpu_pure();
;  return 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [128 x [128 x i32]] zeroinitializer, align 16

define i32 @gpu_pure() nounwind uwtable {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc6, %entry
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %for.inc6 ], [ 0, %entry ]
  %lftr.wideiv5 = trunc i64 %indvars.iv2 to i32
  %exitcond6 = icmp ne i32 %lftr.wideiv5, 128
  br i1 %exitcond6, label %for.body, label %for.end8

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc ], [ 0, %for.body ]
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond, label %for.body3, label %for.end

for.body3:                                        ; preds = %for.cond1
  %tmp = shl nsw i64 %indvars.iv2, 7
  %tmp7 = add nsw i64 %tmp, %indvars.iv
  %arrayidx5 = getelementptr inbounds [128 x [128 x i32]]* @A, i64 0, i64 %indvars.iv2, i64 %indvars.iv
  %tmp8 = trunc i64 %tmp7 to i32
  store i32 %tmp8, i32* %arrayidx5, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body3
  %indvars.iv.next = add i64 %indvars.iv, 1
  br label %for.cond1

for.end:                                          ; preds = %for.cond1
  br label %for.inc6

for.inc6:                                         ; preds = %for.end
  %indvars.iv.next3 = add i64 %indvars.iv2, 1
  br label %for.cond

for.end8:                                         ; preds = %for.cond
  ret i32 0
}

define i32 @main() nounwind uwtable {
entry:
  %call = call i32 @gpu_pure()
  ret i32 0
}

; CHECK:  call void @polly_initDevice
; CHECK:  call void @polly_getPTXModule
; CHECK:  call void @polly_getPTXKernelEntry
; CHECK:  call void @polly_allocateMemoryForHostAndDevice
; CHECK:  call void @polly_setKernelParameters
; CHECK:  call void @polly_startTimerByCudaEvent
; CHECK:  call void @polly_launchKernel
; CHECK:  call void @polly_copyFromDeviceToHost
; CHECK:  call void @polly_stopTimerByCudaEvent
; CHECK:  call void @polly_cleanupGPGPUResources
