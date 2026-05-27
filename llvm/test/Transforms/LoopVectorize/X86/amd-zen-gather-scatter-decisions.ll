; End-to-end loop-vectorize decisions driven by the AMD Zen per-shape
; gather/scatter cost tables (TuningPreferAMDZenGSCost, set on znver4+).
;
; The companion cost-model test
;   llvm/test/Analysis/CostModel/X86/masked-gather-scatter-amd-zen.ll
; pins individual cost numbers; this test pins the resulting vectorizer
; decisions so future cost-model refactors that accidentally re-enable
; harmful gathers (or suppress profitable ones) are caught here.
;
; The three cases below correspond to:
;   1. f64 indirect-load reduction -- gather IS chosen on znver5
;      (the lbm-style win the cost table exists to enable).
;   2. i64 indirect-load reduction -- gather is NOT chosen on znver5;
;      the i64 entry was measured separately and deliberately set above
;      the break-even so vpgatherqq is suppressed for harmful patterns
;      (cf. PR #198850 / libquantum regression).
;   3. Unit-stride load -- the vectorizer must emit a plain wide load
;      (not @llvm.masked.gather) regardless of cost-table values.
;      Regression guard for issue #91370.
;
; RUN: opt < %s -S -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=znver5 | FileCheck %s

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"

; --- Case 1: f64 indirect-load gather IS chosen on znver5 ----------------
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
; CHECK-LABEL: define i64 @i64_indirect_gather_avoided
; CHECK-NOT:   call <{{[0-9]+}} x i64> @llvm.masked.gather.v{{[0-9]+}}i64
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
; CHECK-LABEL: define void @unit_stride_no_gather
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
