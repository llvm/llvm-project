; Verify that the TuningPreferNoScatter feature takes effect when enabled.
; Scatter instructions are generated on CPUs that support them, but not
; when the tuning is on.

; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 -S %s | FileCheck %s --check-prefix=SCATTER
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=c86-4g-m7 -S %s | FileCheck %s --check-prefix=NO-SCATTER
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=c86-4g-m8 -S %s | FileCheck %s --check-prefix=NO-SCATTER

target triple = "x86_64-unknown-linux-gnu"

; SCATTER-LABEL: @scatter_loop(
; SCATTER: call void @llvm.masked.scatter

; NO-SCATTER-LABEL: @scatter_loop(
; NO-SCATTER-NOT: @llvm.masked.scatter

define void @scatter_loop(ptr noalias %in, ptr noalias %out, ptr noalias %index) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i64, ptr %index, i64 %indvars.iv
  %1 = load i64, ptr %0, align 8
  %2 = getelementptr inbounds float, ptr %in, i64 %indvars.iv
  %3 = load float, ptr %2, align 4
  %4 = getelementptr inbounds float, ptr %out, i64 %1
  store float %3, ptr %4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
