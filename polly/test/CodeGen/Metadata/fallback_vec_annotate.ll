; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' -polly-annotate-metadata-vectorize < %s | FileCheck %s
; RUN: opt %loadNPMPolly -S '-passes=polly<no-default-opts>' < %s | FileCheck %s

; Verify vectorization is not disabled when RTC of Polly is false

; CHECK: attributes {{.*}} = { "polly-optimized" }
; CHECK-NOT: {{.*}} = !{!"llvm.loop.vectorize.enable", i32 0}

target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "aarch64-unknown-linux-android10000"

define void @ham(i64 %arg) {
bb:
  br label %bb1

bb1:                                              ; preds = %bb3, %bb
  %phi = phi ptr [ %getelementptr4, %bb3 ], [ null, %bb ]
  br label %bb2

bb2:                                              ; preds = %bb2, %bb1
  %getelementptr = getelementptr i8, ptr %phi, i64 1
  store i8 0, ptr %getelementptr, align 1
  br i1 false, label %bb2, label %bb3

bb3:                                              ; preds = %bb2
  %getelementptr4 = getelementptr i8, ptr %phi, i64 %arg
  br label %bb1
}
