; RUN: opt %loadNPMPolly -polly-process-unprofitable -polly-use-llvm-names -polly-target-throughput-vector-fma=1 -polly-target-latency-vector-fma=8 -polly-target-1st-cache-level-associativity=8 -polly-target-2nd-cache-level-associativity=8 -polly-target-1st-cache-level-size=1024 -polly-target-vector-register-bitwidth=256 -polly-target-2nd-cache-level-size=960 '-passes=polly<no-default-opts;opt-isl>' -S < %s | FileCheck %s
; RUN: opt %loadNPMPolly -polly-process-unprofitable -polly-use-llvm-names -polly-target-throughput-vector-fma=1 -polly-target-latency-vector-fma=8 -polly-target-1st-cache-level-associativity=8 -polly-target-2nd-cache-level-associativity=8 -polly-target-1st-cache-level-size=1024 -polly-target-vector-register-bitwidth=256 -polly-target-2nd-cache-level-size=960 -polly-pattern-matching-max-stack-array-size=0 '-passes=polly<no-default-opts;opt-isl>' -S < %s | FileCheck %s --check-prefix=HEAP
; RUN: opt %loadNPMPolly -polly-process-unprofitable -polly-use-llvm-names -polly-target-throughput-vector-fma=1 -polly-target-latency-vector-fma=8 -polly-target-1st-cache-level-associativity=8 -polly-target-2nd-cache-level-associativity=8 -polly-target-1st-cache-level-size=1024 -polly-target-vector-register-bitwidth=256 -polly-target-2nd-cache-level-size=256 '-passes=polly<no-default-opts;opt-isl>' -S < %s | FileCheck %s --check-prefix=SMALL
;
; With Mc=11 and Mr=4, Packed_A contains two complete micro-tiles and one
; partial micro-tile. The first two runs check its stack and heap allocations.
; With Mc=3, the third run checks a single partial micro-tile.
;
; CHECK-LABEL: define void @matmul(
; CHECK: %Packed_A = alloca [3 x [8 x [4 x double]]]
; SMALL-LABEL: define void @matmul(
; SMALL: %Packed_A = alloca [1 x
; HEAP-LABEL: define void @matmul(
; HEAP: %Packed_A = tail call ptr @malloc(i64 768)
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
  %cp = getelementptr inbounds [8 x double], ptr %C, i64 %i, i64 %j
  br label %inner
inner:
  %k = phi i64 [ 0, %middle ], [ %knext, %inner ]
  %ap = getelementptr inbounds [8 x double], ptr %A, i64 %i, i64 %k
  %bp = getelementptr inbounds [8 x double], ptr %B, i64 %k, i64 %j
  %a = load double, ptr %ap, align 8
  %b = load double, ptr %bp, align 8
  %c = load double, ptr %cp, align 8
  %prod = fmul double %a, %b
  %sum = fadd double %c, %prod
  store double %sum, ptr %cp, align 8
  %knext = add nuw nsw i64 %k, 1
  %kcond = icmp ne i64 %knext, 8
  br i1 %kcond, label %inner, label %middleend
middleend:
  %jnext = add nuw nsw i64 %j, 1
  %jcond = icmp ne i64 %jnext, 8
  br i1 %jcond, label %middle, label %outerend
outerend:
  %inext = add nuw nsw i64 %i, 1
  %icond = icmp ne i64 %inext, 16
  br i1 %icond, label %outer, label %exit
exit:
  ret void
}
