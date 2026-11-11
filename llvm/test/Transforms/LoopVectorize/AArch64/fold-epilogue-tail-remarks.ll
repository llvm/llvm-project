; REQUIRES: asserts

; RUN: opt -S -p loop-vectorize -mattr=+sve -force-vector-width=16 \
; RUN: -epilogue-tail-folding-policy=prefer-fold-tail \
; RUN: -epilogue-vectorization-force-VF=8 --disable-output \
; RUN: -debug-only=loop-vectorize -pass-remarks-analysis=loop-vectorize \
; RUN: %s 2>&1 | FileCheck %s

; RUN: opt -S -p loop-vectorize -debug-only=loop-vectorize \
; RUN: -epilogue-tail-folding-policy=prefer-fold-tail --disable-output \
; RUN: -force-vector-width=4 -epilogue-vectorization-force-VF="vscale x 2" \
; RUN: -pass-remarks-analysis=loop-vectorize \
; RUN: -force-tail-folding-style=data-and-control < %s 2>&1 | FileCheck %s \
; RUN: --check-prefix=CHECK-INVALID-COSTS

target triple = "aarch64-linux-gnu"

define void @uniform_load_store_no_mask(ptr %s, ptr %cfg, i64 %n) {
; CHECK-LABEL: Checking a loop in 'uniform_load_store_no_mask'
; CHECK: LV: epilogue tail-folding is enabled
; CHECK: remark: <unknown>:0:0: Tail-folded epilogue has no active-lane-mask exit, so it may not terminate, can't apply tail-folded epilogue
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %base = load i32, ptr %cfg, align 4
  %next = add nsw i32 %base, 1
  store i32 %next, ptr %s, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define float @fmaxnum(ptr %src, i64 %n) {
; CHECK-LABEL: Checking a loop in 'fmaxnum'
; CHECK: LV: epilogue tail-folding is enabled
; CHECK: remark: <unknown>:0:0: Failed to build tail-folded epilogue VPlan
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %red = phi float [ -1.000000e+03, %entry ], [ %max, %loop ]
  %gep = getelementptr inbounds float, ptr %src, i64 %iv
  %load = load float, ptr %gep, align 4
  %max = call float @llvm.maxnum.f32(float %red, float %load)
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret float %max
}

define void @alloca(ptr %vla, i64 %N) {
; CHECK-INVALID-COSTS-LABEL: Checking a loop in 'alloca'
; CHECK-INVALID-COSTS: LV: epilogue tail-folding is enabled
; CHECK-INVALID-COSTS: remark: <unknown>:0:0: This case of epilogue loop can't be tail-folded - Invalid costs
entry:
  br label %loop

loop:
  %iv = phi i64 [ %iv.next, %loop ], [ 0, %entry ]
  %alloca = alloca i32, align 16
  %arrayidx = getelementptr inbounds ptr, ptr %vla, i64 %iv
  store ptr %alloca, ptr %arrayidx, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %exitcond.not = icmp eq i64 %iv.next, %N
  br i1 %exitcond.not, label %exit, label %loop

exit:
  call void @foo(ptr nonnull %vla)
  ret void
}
declare void @foo(ptr)
