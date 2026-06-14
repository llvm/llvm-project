; REQUIRES: aarch64-registered-target
; REQUIRES: default_triple

;; There shouldn't be a default -mtune.
; RUN: opt < %s -passes=loop-vectorize -S -mtriple=aarch64 | FileCheck %s --check-prefixes=CHECK-NOTUNE

; RUN: opt < %s -passes=loop-vectorize -S -mtriple=aarch64 -mtune=generic | FileCheck %s --check-prefixes=CHECK-TUNE-GENERIC
; RUN: opt < %s -passes=loop-vectorize -S -mtriple=aarch64 -mtune=apple-m5 | FileCheck %s --check-prefixes=CHECK-TUNE-APPLE-M5

;; Check interaction between mcpu and mtune.
; RUN: opt < %s -passes=loop-vectorize -S -mtriple=aarch64 -mcpu=apple-m5 | FileCheck %s --check-prefixes=CHECK-TUNE-APPLE-M5
; RUN: opt < %s -passes=loop-vectorize -S -mtriple=aarch64 -mcpu=apple-m5 -mtune=generic | FileCheck %s --check-prefixes=CHECK-TUNE-GENERIC

;; Test -mtune=help
; RUN: opt -mtriple=aarch64 -mtune=help 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-OPTIMIZE
; RUN: opt -mtriple=aarch64 -mtune=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-OPTIMIZE
; RUN: opt < %s -mtriple=aarch64 -mtune=help 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-OPTIMIZE
; RUN: opt < %s -mtriple=aarch64 -mtune=help -o %t.s 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-OPTIMIZE

;; Using default triple for -mtune=help
; RUN: opt -mtune=help 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-OPTIMIZE
; RUN: opt < %s -mtune=help 2>&1 | FileCheck %s --check-prefixes=CHECK-TUNE-HELP,CHECK-TUNE-HELP-NO-OPTIMIZE

; CHECK-TUNE-HELP: Available CPUs for this target:
; CHECK-TUNE-HELP: Available features for this target:

;; To check we dont optimize the file
; CHECK-TUNE-HELP-NO-OPTIMIZE-NOT: loop:

;; A test case that depends on `FeatureMaxInterleaveFactor4` tuning feature, to enable -mtune verification
;; through codegen effects. Taken from: llvm/test/Transforms/LoopVectorize/max-interleave-factor-debug.ll
define void @loop(ptr noalias %src, ptr noalias %dst, i64 %n) {
; CHECK-NOTUNE-LABEL: define void @loop(
; CHECK-NOTUNE:       [[VECTOR_BODY:.*]]:
; CHECK-NOTUNE:         [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr
; CHECK-NOTUNE-NEXT:    [[WIDE_LOAD1:%.*]] = load <4 x i32>, ptr
; CHECK-NOTUNE:         store <4 x i32> [[WIDE_LOAD]], ptr
; CHECK-NOTUNE-NEXT:    store <4 x i32> [[WIDE_LOAD1]], ptr
; CHECK-NOTUNE-NEXT:    [[INDEX_NEXT:%.*]] = add nuw i64 [[INDEX:%.*]], 8
;
; CHECK-TUNE-GENERIC-LABEL: define void @loop(
; CHECK-TUNE-GENERIC:       [[VECTOR_BODY:.*]]:
; CHECK-TUNE-GENERIC:         [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr
; CHECK-TUNE-GENERIC-NEXT:    [[WIDE_LOAD1:%.*]] = load <4 x i32>, ptr
; CHECK-TUNE-GENERIC:         store <4 x i32> [[WIDE_LOAD]], ptr
; CHECK-TUNE-GENERIC-NEXT:    store <4 x i32> [[WIDE_LOAD1]], ptr
; CHECK-TUNE-GENERIC-NEXT:    [[INDEX_NEXT:%.*]] = add nuw i64 [[INDEX:%.*]], 8
;
; CHECK-TUNE-APPLE-M5-LABEL: define void @loop(
; CHECK-TUNE-APPLE-M5:       [[VECTOR_BODY:.*]]:
; CHECK-TUNE-APPLE-M5:         [[WIDE_LOAD:%.*]] = load <4 x i32>, ptr
; CHECK-TUNE-APPLE-M5-NEXT:    [[WIDE_LOAD2:%.*]] = load <4 x i32>, ptr
; CHECK-TUNE-APPLE-M5-NEXT:    [[WIDE_LOAD3:%.*]] = load <4 x i32>, ptr
; CHECK-TUNE-APPLE-M5-NEXT:    [[WIDE_LOAD4:%.*]] = load <4 x i32>, ptr
; CHECK-TUNE-APPLE-M5:         store <4 x i32> [[WIDE_LOAD]], ptr
; CHECK-TUNE-APPLE-M5-NEXT:    store <4 x i32> [[WIDE_LOAD2]], ptr
; CHECK-TUNE-APPLE-M5-NEXT:    store <4 x i32> [[WIDE_LOAD3]], ptr
; CHECK-TUNE-APPLE-M5-NEXT:    store <4 x i32> [[WIDE_LOAD4]], ptr
; CHECK-TUNE-APPLE-M5-NEXT:    [[INDEX_NEXT:%.*]] = add nuw i64 [[INDEX:%.*]], 16
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %p = getelementptr i32, ptr %src, i64 %iv
  %v = load i32, ptr %p, align 4
  %q = getelementptr i32, ptr %dst, i64 %iv
  store i32 %v, ptr %q, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %cond = icmp ne i64 %iv.next, %n
  br i1 %cond, label %loop, label %exit

exit:
  ret void
}
