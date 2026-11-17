; RUN: opt %loadNPMPolly -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-use-llvm-names -plugin-arg=Polly,-polly-target-throughput-vector-fma=1 -plugin-arg=Polly,-polly-target-latency-vector-fma=8 -plugin-arg=Polly,-polly-target-1st-cache-level-associativity=8 -plugin-arg=Polly,-polly-target-2nd-cache-level-associativity=8 -plugin-arg=Polly,-polly-target-1st-cache-level-size=32768 -plugin-arg=Polly,-polly-target-vector-register-bitwidth=256 -plugin-arg=Polly,-polly-target-2nd-cache-level-size=262144 '-passes=polly<no-default-opts;opt-isl>' -S < %s | FileCheck %s --check-prefix=STACK
; RUN: opt %loadNPMPolly -plugin-arg=Polly,-polly-process-unprofitable -plugin-arg=Polly,-polly-use-llvm-names -plugin-arg=Polly,-polly-target-throughput-vector-fma=1 -plugin-arg=Polly,-polly-target-latency-vector-fma=8 -plugin-arg=Polly,-polly-target-1st-cache-level-associativity=8 -plugin-arg=Polly,-polly-target-2nd-cache-level-associativity=8 -plugin-arg=Polly,-polly-target-1st-cache-level-size=32768 -plugin-arg=Polly,-polly-target-vector-register-bitwidth=256 -plugin-arg=Polly,-polly-target-2nd-cache-level-size=262144 -plugin-arg=Polly,-polly-pattern-matching-max-stack-array-size=512 '-passes=polly<no-default-opts;opt-isl>' -S < %s | FileCheck %s --check-prefix=HEAP
;
; Both products use the same Packed_A and Packed_B buffers. Their final shapes
; contain the elementwise maximum of the two products' packing bounds.
; STACK-LABEL: define void @two_matmuls(
; STACK: %Packed_B = alloca [2 x [7 x [8 x double]]]
; STACK-NEXT: %Packed_A = alloca [1 x [7 x [3 x double]]]
; STACK: br i1 true, label %polly.start
; STACK-NOT: call ptr @malloc
; STACK: ret void
;
; The enlarged Packed_B crosses the 512-byte stack allocation threshold.
; HEAP-LABEL: define void @two_matmuls(
; HEAP: %Packed_A = alloca [1 x [7 x [3 x double]]]
; HEAP: br i1 true, label %polly.start
; HEAP: %Packed_B = tail call ptr @malloc(i64 896)
; HEAP: tail call void @free(ptr %Packed_B)
; HEAP: }
;
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"
define void @two_matmuls(ptr noalias %A, ptr noalias %B, ptr noalias %Tmp, ptr noalias %D, ptr noalias %E) {
entry:
  br label %outer0
outer0:
  %i0 = phi i64 [ 0, %entry ], [ %inext0, %outerend0 ]
  br label %middle0
middle0:
  %j0 = phi i64 [ 0, %outer0 ], [ %jnext0, %middleend0 ]
  %cp0 = getelementptr inbounds [5 x double], ptr %Tmp, i64 %i0, i64 %j0
  br label %inner0
inner0:
  %k0 = phi i64 [ 0, %middle0 ], [ %knext0, %inner0 ]
  %ap0 = getelementptr inbounds [7 x double], ptr %A, i64 %i0, i64 %k0
  %bp0 = getelementptr inbounds [5 x double], ptr %B, i64 %k0, i64 %j0
  %a0 = load double, ptr %ap0, align 8
  %b0 = load double, ptr %bp0, align 8
  %c0 = load double, ptr %cp0, align 8
  %prod0 = fmul double %a0, %b0
  %sum0 = fadd double %c0, %prod0
  store double %sum0, ptr %cp0, align 8
  %knext0 = add nuw nsw i64 %k0, 1
  %kcond0 = icmp ne i64 %knext0, 7
  br i1 %kcond0, label %inner0, label %middleend0
middleend0:
  %jnext0 = add nuw nsw i64 %j0, 1
  %jcond0 = icmp ne i64 %jnext0, 5
  br i1 %jcond0, label %middle0, label %outerend0
outerend0:
  %inext0 = add nuw nsw i64 %i0, 1
  %icond0 = icmp ne i64 %inext0, 3
  br i1 %icond0, label %outer0, label %next
next:
  br label %outer1
outer1:
  %i1 = phi i64 [ 0, %next ], [ %inext1, %outerend1 ]
  br label %middle1
middle1:
  %j1 = phi i64 [ 0, %outer1 ], [ %jnext1, %middleend1 ]
  %cp1 = getelementptr inbounds [9 x double], ptr %E, i64 %i1, i64 %j1
  br label %inner1
inner1:
  %k1 = phi i64 [ 0, %middle1 ], [ %knext1, %inner1 ]
  %ap1 = getelementptr inbounds [5 x double], ptr %Tmp, i64 %i1, i64 %k1
  %bp1 = getelementptr inbounds [9 x double], ptr %D, i64 %k1, i64 %j1
  %a1 = load double, ptr %ap1, align 8
  %b1 = load double, ptr %bp1, align 8
  %c1 = load double, ptr %cp1, align 8
  %prod1 = fmul double %a1, %b1
  %sum1 = fadd double %c1, %prod1
  store double %sum1, ptr %cp1, align 8
  %knext1 = add nuw nsw i64 %k1, 1
  %kcond1 = icmp ne i64 %knext1, 5
  br i1 %kcond1, label %inner1, label %middleend1
middleend1:
  %jnext1 = add nuw nsw i64 %j1, 1
  %jcond1 = icmp ne i64 %jnext1, 9
  br i1 %jcond1, label %middle1, label %outerend1
outerend1:
  %inext1 = add nuw nsw i64 %i1, 1
  %icond1 = icmp ne i64 %inext1, 3
  br i1 %icond1, label %outer1, label %exit
exit:
  ret void
}
