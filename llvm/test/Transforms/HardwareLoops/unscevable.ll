; RUN: opt -passes=hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -S %s -o - | FileCheck %s
; RUN: opt -passes=hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-hardware-loop-phi=true -S %s -o - | FileCheck %s
; RUN: opt -passes=hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-nested-hardware-loop=true -S %s -o - | FileCheck %s

; CHECK-LABEL: float_counter
; CHECK-NOT: set.loop.iterations
; CHECK-NOT: loop.decrement
define void @float_counter(ptr nocapture %A, float %N) {
entry:
  %cmp6 = fcmp ogt float %N, 0.000000e+00
  br i1 %cmp6, label %while.body, label %while.end

while.body:
  %i.07 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, ptr %A, i32 %i.07
  store i32 %i.07, ptr %arrayidx, align 4
  %inc = add i32 %i.07, 1
  %conv = uitofp i32 %inc to float
  %cmp = fcmp olt float %conv, %N
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

; CHECK-LABEL: variant_counter
; CHECK-NOT: set.loop.iterations
; CHECK-NOT: loop.decrement
define void @variant_counter(ptr nocapture %A, ptr nocapture readonly %B) {
entry:
  %0 = load i32, ptr %B, align 4
  %cmp7 = icmp eq i32 %0, 0
  br i1 %cmp7, label %while.end, label %while.body

while.body:
  %i.08 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %arrayidx1 = getelementptr inbounds i32, ptr %A, i32 %i.08
  store i32 %i.08, ptr %arrayidx1, align 4
  %inc = add nuw i32 %i.08, 1
  %arrayidx = getelementptr inbounds i32, ptr %B, i32 %inc
  %1 = load i32, ptr %arrayidx, align 4
  %cmp = icmp ult i32 %inc, %1
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}

; CHECK-LABEL: variant_counter2
; CHECK-NOT: set.loop.iterations
; CHECK-NOT: loop.decrement
define void @variant_counter2(ptr, ptr, ptr) {
  %4 = icmp eq ptr %0, %1
  br i1 %4, label %9, label %5

5:                                                ; preds = %3
  %6 = getelementptr inbounds i64, ptr %2, i64 1
  %7 = load i64, ptr %6, align 8
  br label %10

8:                                                ; preds = %10
  store i64 %14, ptr %6, align 8
  br label %9

9:                                                ; preds = %8, %3
  ret void

10:                                               ; preds = %5, %10
  %11 = phi i64 [ %7, %5 ], [ %14, %10 ]
  %12 = phi i32 [ 0, %5 ], [ %15, %10 ]
  %13 = phi ptr [ %0, %5 ], [ %16, %10 ]
  %14 = shl nsw i64 %11, 4
  %15 = add nuw nsw i32 %12, 1
  %16 = getelementptr inbounds i8, ptr %13, i64 1
  %17 = icmp ugt i32 %12, 14
  %18 = icmp eq ptr %16, %1
  %19 = or i1 %18, %17
  br i1 %19, label %8, label %10
}
