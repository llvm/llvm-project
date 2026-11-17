; RUN: opt %loadNPMPolly -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-use-llvm-names -plugin-arg=Polly,-polly-target-throughput-vector-fma=1 -plugin-arg=Polly,-polly-target-latency-vector-fma=8 -plugin-arg=Polly,-polly-target-1st-cache-level-associativity=8 -plugin-arg=Polly,-polly-target-2nd-cache-level-associativity=8 -plugin-arg=Polly,-polly-target-1st-cache-level-size=32768 -plugin-arg=Polly,-polly-target-vector-register-bitwidth=256 -plugin-arg=Polly,-polly-target-2nd-cache-level-size=262144 '-passes=polly<no-default-opts;opt-isl>' -S < %s | FileCheck %s
;
; Packed_A and Packed_B contain the 3x7 and 7x5 accessed elements, respectively.
; Both allocations fit below the stack allocation threshold.
; CHECK-LABEL: define void @matmul(
; CHECK: %Packed_B = alloca [1 x [7 x [5 x double]]]
; CHECK-NEXT: %Packed_A = alloca [1 x [7 x [3 x double]]]
; CHECK-NOT: call ptr @malloc
; CHECK: ret void
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define void @matmul(ptr noalias %A, ptr noalias %B, ptr noalias %C) {
entry:
  br label %outer
outer:
  %i = phi i64 [ 0, %entry ], [ %inext, %outerend ]
  br label %middle
middle:
  %j = phi i64 [ 0, %outer ], [ %jnext, %middleend ]
  %cp = getelementptr inbounds [5 x double], ptr %C, i64 %i, i64 %j
  br label %inner
inner:
  %k = phi i64 [ 0, %middle ], [ %knext, %inner ]
  %ap = getelementptr inbounds [7 x double], ptr %A, i64 %i, i64 %k
  %bp = getelementptr inbounds [5 x double], ptr %B, i64 %k, i64 %j
  %a = load double, ptr %ap, align 8
  %b = load double, ptr %bp, align 8
  %c = load double, ptr %cp, align 8
  %prod = fmul double %a, %b
  %sum = fadd double %c, %prod
  store double %sum, ptr %cp, align 8
  %knext = add nuw nsw i64 %k, 1
  %kcond = icmp ne i64 %knext, 7
  br i1 %kcond, label %inner, label %middleend
middleend:
  %jnext = add nuw nsw i64 %j, 1
  %jcond = icmp ne i64 %jnext, 5
  br i1 %jcond, label %middle, label %outerend
outerend:
  %inext = add nuw nsw i64 %i, 1
  %icond = icmp ne i64 %inext, 3
  br i1 %icond, label %outer, label %exit
exit:
  ret void
}
