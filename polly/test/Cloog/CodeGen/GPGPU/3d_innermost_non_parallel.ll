; RUN: opt %loadPolly -basicaa -polly-import-jscop -polly-import-jscop-dir=%S -polly-import-jscop-postfix=transformed+gpu -enable-polly-gpgpu -polly-gpgpu-triple=nvptx64-unknown-unknown -polly-codegen < %s -S | FileCheck %s

;int A[128][128];
;
;int gpu_no_pure() {
;  int i,j,k;
;
;  for(i = 0; i < 128; i++)
;    for(j = 0; j < 128; j++)
;      for(k = 0; k < 256; k++)
;        A[i][j] += i*123/(k+1)+5-j*k-123;
;
;  return 0;
;}
;
;int main() {
;  int b = gpu_no_pure();
;  return 0;
;}

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@A = common global [128 x [128 x i32]] zeroinitializer, align 16

define i32 @gpu_no_pure() nounwind uwtable {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.inc16, %entry
  %indvars.iv2 = phi i64 [ %indvars.iv.next3, %for.inc16 ], [ 0, %entry ]
  %lftr.wideiv5 = trunc i64 %indvars.iv2 to i32
  %exitcond6 = icmp ne i32 %lftr.wideiv5, 128
  br i1 %exitcond6, label %for.body, label %for.end18

for.body:                                         ; preds = %for.cond
  br label %for.cond1

for.cond1:                                        ; preds = %for.inc13, %for.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.inc13 ], [ 0, %for.body ]
  %lftr.wideiv = trunc i64 %indvars.iv to i32
  %exitcond1 = icmp ne i32 %lftr.wideiv, 128
  br i1 %exitcond1, label %for.body3, label %for.end15

for.body3:                                        ; preds = %for.cond1
  br label %for.cond4

for.cond4:                                        ; preds = %for.inc, %for.body3
  %k.0 = phi i32 [ 0, %for.body3 ], [ %inc, %for.inc ]
  %exitcond = icmp ne i32 %k.0, 256
  br i1 %exitcond, label %for.body6, label %for.end

for.body6:                                        ; preds = %for.cond4
  %tmp = mul nsw i64 %indvars.iv2, 123
  %add = add nsw i32 %k.0, 1
  %tmp7 = trunc i64 %tmp to i32
  %div = sdiv i32 %tmp7, %add
  %add7 = add nsw i32 %div, 5
  %tmp8 = trunc i64 %indvars.iv to i32
  %mul8 = mul nsw i32 %tmp8, %k.0
  %sub = sub nsw i32 %add7, %mul8
  %sub9 = add nsw i32 %sub, -123
  %arrayidx11 = getelementptr inbounds [128 x [128 x i32]]* @A, i64 0, i64 %indvars.iv2, i64 %indvars.iv
  %tmp9 = load i32* %arrayidx11, align 4
  %add12 = add nsw i32 %tmp9, %sub9
  store i32 %add12, i32* %arrayidx11, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body6
  %inc = add nsw i32 %k.0, 1
  br label %for.cond4

for.end:                                          ; preds = %for.cond4
  br label %for.inc13

for.inc13:                                        ; preds = %for.end
  %indvars.iv.next = add i64 %indvars.iv, 1
  br label %for.cond1

for.end15:                                        ; preds = %for.cond1
  br label %for.inc16

for.inc16:                                        ; preds = %for.end15
  %indvars.iv.next3 = add i64 %indvars.iv2, 1
  br label %for.cond

for.end18:                                        ; preds = %for.cond
  ret i32 0
}

define i32 @main() nounwind uwtable {
entry:
  %call = call i32 @gpu_no_pure()
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
