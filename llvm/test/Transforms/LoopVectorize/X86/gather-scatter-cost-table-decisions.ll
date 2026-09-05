; End-to-end loop-vectorize decisions driven by the per-shape gather/scatter
; cost tables (TuningPreferGSCostTable, set on znver4+). The companion
; cost-model test masked-gather-scatter-cost-table.ll pins the individual cost
; numbers; this test pins the resulting vectorizer decisions so cost-model
; refactors that re-enable harmful gathers (or suppress profitable ones) are
; caught here.
;
; Each case is run on znver5 (per-shape tables) and on skylake-avx512 (generic
; flat overhead). Where the two disagree the decision is table-driven, and the
; SKX run is what stops the znver5 CHECK-NOTs from passing vacuously.
;
; The five cases below:
;   1. f64 indirect-load reduction -- gather IS chosen, on both.
;   2. i64 indirect-load reduction -- gather is NOT chosen on znver5, but IS on
;      SKX (the i64 entry is set above the break-even to suppress vpgatherqq
;      for harmful patterns, cf. PR llvm#198850).
;   3. Unit-stride load -- must stay a plain wide load, not a gather, on both.
;      Regression guard for issue llvm#91370.
;   4. f64 indirect store -- scatter IS chosen, on both.
;   5. i64 indirect store -- scatter is NOT chosen on znver5, but IS on SKX.
;
; RUN: opt < %s -S -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -mcpu=znver5 | FileCheck %s --check-prefixes=CHECK,ZNVER5
; RUN: opt < %s -S -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu \
; RUN:   -mcpu=skylake-avx512 | FileCheck %s --check-prefixes=CHECK,SKX

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; --- Case 1: f64 indirect-load gather IS chosen ---------------------------
; CHECK-LABEL: define double @f64_indirect_gather_chosen
; CHECK:       call <{{[0-9]+}} x double> @llvm.masked.gather.v{{[0-9]+}}f64
define double @f64_indirect_gather_chosen(ptr noundef readonly %data, ptr noundef readonly %idx, i32 noundef %n) {
entry:
  %cmp = icmp ugt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi double [ 0.0, %entry ], [ %acc.next, %loop ]
  %idx.gep = getelementptr inbounds i32, ptr %idx, i32 %i
  %idx.val = load i32, ptr %idx.gep, align 4
  %idx.sext = sext i32 %idx.val to i64
  %data.gep = getelementptr inbounds double, ptr %data, i64 %idx.sext
  %data.val = load double, ptr %data.gep, align 8
  %acc.next = fadd fast double %acc, %data.val
  %inc = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  %ret = phi double [ 0.0, %entry ], [ %acc.next, %loop ]
  ret double %ret
}

; --- Case 2: i64 indirect-load gather is NOT chosen on znver5 ------------
; The positive CHECK on vector.body distinguishes "vectorized without a
; gather" from "did not vectorize at all" -- without it, a future regression
; that fails to vectorize the loop entirely would pass CHECK-NOT vacuously.
; SKX takes the gather here, so the two runs pin the cost difference itself.
; CHECK-LABEL: define i64 @i64_indirect_gather_avoided
; CHECK:       vector.body
; ZNVER5-NOT:  call <{{[0-9]+}} x i64> @llvm.masked.gather.v{{[0-9]+}}i64
; SKX:         call <{{[0-9]+}} x i64> @llvm.masked.gather.v{{[0-9]+}}i64
define i64 @i64_indirect_gather_avoided(ptr noundef readonly %data, ptr noundef readonly %idx, i32 noundef %n) {
entry:
  %cmp = icmp ugt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %acc = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  %idx.gep = getelementptr inbounds i64, ptr %idx, i32 %i
  %idx.val = load i64, ptr %idx.gep, align 8
  %data.gep = getelementptr inbounds i64, ptr %data, i64 %idx.val
  %data.val = load i64, ptr %data.gep, align 8
  %acc.next = add i64 %acc, %data.val
  %inc = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  %ret = phi i64 [ 0, %entry ], [ %acc.next, %loop ]
  ret i64 %ret
}

; --- Case 3: unit-stride load must NOT become a gather (#91370 guard) -----
; Same vector.body anchor as Case 2: ensures the loop did vectorize (to a
; wide load) rather than failing to vectorize entirely.
; CHECK-LABEL: define void @unit_stride_no_gather
; CHECK:       vector.body
; CHECK-NOT:   call <{{[0-9]+}} x double> @llvm.masked.gather
define void @unit_stride_no_gather(ptr noundef writeonly %out, ptr noundef readonly %in, i32 noundef %n) {
entry:
  %cmp = icmp ugt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %in.gep = getelementptr inbounds double, ptr %in, i32 %i
  %in.val = load double, ptr %in.gep, align 8
  %mul = fmul fast double %in.val, 2.000000e+00
  %out.gep = getelementptr inbounds double, ptr %out, i32 %i
  store double %mul, ptr %out.gep, align 8
  %inc = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; --- Case 4: f64 indirect-store scatter IS chosen ------------------------
; CHECK-LABEL: define void @f64_indirect_scatter_chosen
; CHECK:       call void @llvm.masked.scatter.v{{[0-9]+}}f64
define void @f64_indirect_scatter_chosen(ptr noalias noundef writeonly %data, ptr noalias noundef readonly %idx, ptr noalias noundef readonly %src, i32 noundef %n) {
entry:
  %cmp = icmp ugt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %idx.gep = getelementptr inbounds i32, ptr %idx, i32 %i
  %idx.val = load i32, ptr %idx.gep, align 4
  %idx.sext = sext i32 %idx.val to i64
  %src.gep = getelementptr inbounds double, ptr %src, i32 %i
  %src.val = load double, ptr %src.gep, align 8
  %data.gep = getelementptr inbounds double, ptr %data, i64 %idx.sext
  store double %src.val, ptr %data.gep, align 8
  %inc = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

; --- Case 5: i64 indirect-store scatter is NOT chosen on znver5 ----------
; Same vector.body anchor as Case 2, and the same SKX contrast: SKX scatters
; here, so the znver5 CHECK-NOT is pinning a cost decision rather than an
; unconditional refusal to scatter.
; CHECK-LABEL: define void @i64_indirect_scatter_avoided
; CHECK:       vector.body
; ZNVER5-NOT:  call void @llvm.masked.scatter.v{{[0-9]+}}i64
; SKX:         call void @llvm.masked.scatter.v{{[0-9]+}}i64
define void @i64_indirect_scatter_avoided(ptr noalias noundef writeonly %data, ptr noalias noundef readonly %idx, ptr noalias noundef readonly %src, i32 noundef %n) {
entry:
  %cmp = icmp ugt i32 %n, 0
  br i1 %cmp, label %loop, label %exit

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %loop ]
  %idx.gep = getelementptr inbounds i64, ptr %idx, i32 %i
  %idx.val = load i64, ptr %idx.gep, align 8
  %src.gep = getelementptr inbounds i64, ptr %src, i32 %i
  %src.val = load i64, ptr %src.gep, align 8
  %data.gep = getelementptr inbounds i64, ptr %data, i64 %idx.val
  store i64 %src.val, ptr %data.gep, align 8
  %inc = add nuw nsw i32 %i, 1
  %done = icmp eq i32 %inc, %n
  br i1 %done, label %exit, label %loop

exit:
  ret void
}
