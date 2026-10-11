; Verify that the TuningPreferNoGather feature takes effect when enabled.
; Gather instructions are generated on CPUs that support them, but not
; when the tuning is on.

; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=skylake-avx512 -S %s | FileCheck %s --check-prefix=GATHER
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=c86-4g-m7 -S %s | FileCheck %s --check-prefix=NO-GATHER
; RUN: opt -passes=loop-vectorize -mtriple=x86_64-unknown-linux-gnu -mcpu=c86-4g-m8 -S %s | FileCheck %s --check-prefix=NO-GATHER

target triple = "x86_64-unknown-linux-gnu"

; GATHER-LABEL: @gather_loop(
; GATHER: call <{{.*}} x float> @llvm.masked.gather

; NO-GATHER-LABEL: @gather_loop(
; NO-GATHER-NOT: @llvm.masked.gather

define void @gather_loop(ptr noalias %in, ptr noalias %out, ptr noalias %index) {
entry:
  br label %for.body

for.body:
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %0 = getelementptr inbounds i64, ptr %index, i64 %indvars.iv
  %1 = load i64, ptr %0, align 8
  %2 = getelementptr inbounds float, ptr %in, i64 %1
  %3 = load float, ptr %2, align 4
  %4 = getelementptr inbounds float, ptr %out, i64 %indvars.iv
  store float %3, ptr %4, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1024
  br i1 %exitcond, label %for.end, label %for.body

for.end:
  ret void
}
