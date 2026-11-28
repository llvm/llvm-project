; RUN: opt %loadNPMPolly -polly-process-unprofitable -polly-use-llvm-names -polly-target-throughput-vector-fma=1 -polly-target-latency-vector-fma=8 -polly-target-1st-cache-level-associativity=8 -polly-target-2nd-cache-level-associativity=8 -polly-target-1st-cache-level-size=64 -polly-target-vector-register-bitwidth=256 -polly-target-2nd-cache-level-size=262144 '-passes=polly-custom<opt-isl;ast>' -polly-print-ast -disable-output < %s | FileCheck %s
; RUN: opt %loadNPMPolly -polly-process-unprofitable -polly-use-llvm-names -polly-target-throughput-vector-fma=1 -polly-target-latency-vector-fma=8 -polly-target-1st-cache-level-associativity=8 -polly-target-2nd-cache-level-associativity=8 -polly-target-1st-cache-level-size=32768 -polly-target-vector-register-bitwidth=256 -polly-target-2nd-cache-level-size=64 '-passes=polly-custom<opt-isl;ast>' -polly-print-ast -disable-output < %s | FileCheck %s
;
; The first run produces Kc=0 and the second Mc=0. In both cases the fallback
; schedule applies register tiling without cache tiling or packing.
;
; CHECK-NOT: CopyStmt
; CHECK: // Register tiling - Tiles
; CHECK: // Register tiling - Points
; CHECK: Stmt_inner(
; CHECK-NOT: CopyStmt
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
