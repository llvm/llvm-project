; REQUIRES: asserts
;
; The Cortext-A57 machine model will avoid scheduling load instructions in
; succession because loads on the A57 have a latency of 4 cycles and they all
; issue to the same pipeline. Instead, it will move other instructions between
; the loads to avoid unnecessary stalls. The generic machine model schedules 4
; loads consecutively for this case and will cause stalls.
;
; RUN: llc < %s -mtriple=arm64-linux-gnu -mcpu=cortex-a57 -enable-misched -verify-misched -debug-only=machine-scheduler -o - 2>&1 > /dev/null | FileCheck %s
; CHECK: ********** MI Scheduling **********
; CHECK: main:%bb.2
; CHECK: LDR
; CHECK: Latency : 4
; CHECK: *** Final schedule for %bb.2 ***
; CHECK: LDR
; CHECK: LDR
; CHECK-NOT: LDR
; CHECK: {{.*}}
; CHECK: ********** MI Scheduling **********

@main.x = private unnamed_addr constant [8 x i32] [i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1], align 4
@main.y = private unnamed_addr constant [8 x i32] [i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2, i32 2], align 4

; Function Attrs: nounwind
define i32 @main() #0 {
entry:
  %retval = alloca i32, align 4
  %x = alloca [8 x i32], align 4
  %y = alloca [8 x i32], align 4
  %i = alloca i32, align 4
  %xx = alloca i32, align 4
  %yy = alloca i32, align 4
  store i32 0, ptr %retval
  %0 = bitcast ptr %x to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %0, ptr align 4 @main.x, i64 32, i1 false)
  %1 = bitcast ptr %y to ptr
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %1, ptr align 4 @main.y, i64 32, i1 false)
  store i32 0, ptr %xx, align 4
  store i32 0, ptr %yy, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %2 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %2, 8
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load i32, ptr %yy, align 4
  %4 = load i32, ptr %i, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds [8 x i32], ptr %x, i32 0, i64 %idxprom
  %5 = load i32, ptr %arrayidx, align 4
  %add = add nsw i32 %5, 1
  store i32 %add, ptr %xx, align 4
  %6 = load i32, ptr %xx, align 4
  %add1 = add nsw i32 %6, 12
  store i32 %add1, ptr %xx, align 4
  %7 = load i32, ptr %xx, align 4
  %add2 = add nsw i32 %7, 23
  store i32 %add2, ptr %xx, align 4
  %8 = load i32, ptr %xx, align 4
  %add3 = add nsw i32 %8, 34
  store i32 %add3, ptr %xx, align 4
  %9 = load i32, ptr %i, align 4
  %idxprom4 = sext i32 %9 to i64
  %arrayidx5 = getelementptr inbounds [8 x i32], ptr %y, i32 0, i64 %idxprom4
  %10 = load i32, ptr %arrayidx5, align 4

  %add4 = add nsw i32 %9, %add
  %add5 = add nsw i32 %10, %add1
  %add6 = add nsw i32 %add4, %add5

  %add7 = add nsw i32 %9, %add3
  %add8 = add nsw i32 %10, %add4
  %add9 = add nsw i32 %add7, %add8

  %add10 = add nsw i32 %9, %add6
  %add11 = add nsw i32 %10, %add7
  %add12 = add nsw i32 %add10, %add11

  %add13 = add nsw i32 %9, %add9
  %add14 = add nsw i32 %10, %add10
  %add15 = add nsw i32 %add13, %add14

  store i32 %add15, ptr %xx, align 4

  %div = sdiv i32 %4, %5

  store i32 %div, ptr %yy, align 4

  br label %for.inc

for.inc:                                          ; preds = %for.body
  %11 = load i32, ptr %i, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %12 = load i32, ptr %xx, align 4
  %13 = load i32, ptr %yy, align 4
  %add67 = add nsw i32 %12, %13
  ret i32 %add67
}


; Function Attrs: nounwind
declare void @llvm.memcpy.p0.p0.i64(ptr nocapture, ptr nocapture readonly, i64, i1) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind }
