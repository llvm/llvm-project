; REQUIRES: asserts
; RUN: not --crash opt -passes=loop-vectorize -S %s 2>&1 | FileCheck %s --check-prefix=ASSERTION

; Tests a partial reduction with an fadd->fsub chain.
; There's an assertion preventing this type of partial reduction from
; being generated as the current codegen for this case is incorrect.

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-none-unknown-elf"

; ASSERTION: (Chain.RK != RecurKind::FAddChainWithSubs)
define float @fadd_fsub_reduction(float %startval, ptr %src1, ptr %src2, ptr %src3) #0 {
entry:
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %accum = phi float [ %startval, %entry ], [ %sub, %loop ]
  %src1.gep = getelementptr half, ptr %src1, i32 %iv
  %src1.load = load half, ptr %src1.gep, align 4
  %src1.load.ext = fpext half %src1.load to float
  %src2.gep = getelementptr half, ptr %src2, i32 %iv
  %src2.load = load half, ptr %src2.gep, align 4
  %src2.load.ext = fpext half %src2.load to float
  %src3.gep = getelementptr half, ptr %src3, i32 %iv
  %src3.load = load half, ptr %src3.gep, align 4
  %src3.load.ext = fpext half %src3.load to float
  %mul1 = fmul reassoc contract float %src1.load.ext, %src2.load.ext
  %add = fadd reassoc contract float %accum, %mul1
  %mul2 = fmul reassoc contract float %src3.load.ext, %src1.load.ext
  %sub = fsub reassoc contract float %add, %mul2
  %iv.next = add i32 %iv, 1
  %exitcond.not = icmp eq i32 %iv, 1024
  br i1 %exitcond.not, label %exit, label %loop

exit:
  ret float %sub
}

attributes #0 = { vscale_range(1,16) "target-features"="+sve2" }