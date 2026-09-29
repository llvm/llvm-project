; RUN: llc -mtriple=powerpc64-unknown-linux-gnu < %s -verify-machineinstrs | FileCheck %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

%struct.__jmp_buf_tag.1.15.17.21.25.49.53.55 = type { [64 x i64], i32, %struct.__sigset_t.0.14.16.20.24.48.52.54, [8 x i8] }
%struct.__sigset_t.0.14.16.20.24.48.52.54 = type { [16 x i64] }

@env_sigill = external global [1 x %struct.__jmp_buf_tag.1.15.17.21.25.49.53.55], align 16

; CHECK-LABEL: @main
; CHECK-NOT: mtctr

; Function Attrs: nounwind
define void @main() #0 {
entry:
  br i1 undef, label %return, label %if.end

if.end:                                           ; preds = %entry
  br i1 undef, label %for.body.lr.ph, label %for.end.thread

for.end.thread:                                   ; preds = %if.end
  br label %return

for.body.lr.ph:                                   ; preds = %if.end
  br label %for.body

for.cond:                                         ; preds = %for.body
  %cmp2 = icmp slt i32 %inc, undef
  br i1 %cmp2, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond, %for.body.lr.ph
  %i.032 = phi i32 [ 0, %for.body.lr.ph ], [ %inc, %for.cond ]
  %0 = call i32 @llvm.eh.sjlj.setjmp(ptr @env_sigill)
  %inc = add nsw i32 %i.032, 1
  br i1 false, label %if.else, label %for.cond

if.else:                                          ; preds = %for.body
  unreachable

for.end:                                          ; preds = %for.cond
  unreachable

return:                                           ; preds = %for.end.thread, %entry
  ret void
}

; Function Attrs: nounwind
declare i32 @llvm.eh.sjlj.setjmp(ptr) #0

attributes #0 = { nounwind }
