; RUN: opt -hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -S %s -o - | FileCheck %s
; RUN: opt -hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-hardware-loop-phi=true -S %s -o - | FileCheck %s
; RUN: opt -hardware-loops -force-hardware-loops=true -hardware-loop-decrement=1 -hardware-loop-counter-bitwidth=32 -force-nested-hardware-loop=true -S %s -o - | FileCheck %s

; CHECK-LABEL: float_counter
; CHECK-NOT: set.loop.iterations
; CHECK-NOT: loop.decrement
define void @float_counter(i32* nocapture %A, float %N) {
entry:
  %cmp6 = fcmp ogt float %N, 0.000000e+00
  br i1 %cmp6, label %while.body, label %while.end

while.body:
  %i.07 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i32 %i.07
  store i32 %i.07, i32* %arrayidx, align 4
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
define void @variant_counter(i32* nocapture %A, i32* nocapture readonly %B) {
entry:
  %0 = load i32, i32* %B, align 4
  %cmp7 = icmp eq i32 %0, 0
  br i1 %cmp7, label %while.end, label %while.body

while.body:
  %i.08 = phi i32 [ %inc, %while.body ], [ 0, %entry ]
  %arrayidx1 = getelementptr inbounds i32, i32* %A, i32 %i.08
  store i32 %i.08, i32* %arrayidx1, align 4
  %inc = add nuw i32 %i.08, 1
  %arrayidx = getelementptr inbounds i32, i32* %B, i32 %inc
  %1 = load i32, i32* %arrayidx, align 4
  %cmp = icmp ult i32 %inc, %1
  br i1 %cmp, label %while.body, label %while.end

while.end:
  ret void
}
