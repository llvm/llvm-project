; RUN: opt -disable-output -debug-pass-manager -passes-ep-late-loop-optimizations=no-op-loop -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-loop-optimizer-end=no-op-loop -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-scalar-optimizer-late=no-op-function -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-cgscc-optimizer-late=no-op-cgscc -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-vectorizer-start=no-op-function -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-vectorizer-end=no-op-function -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-pipeline-start=no-op-module -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-pipeline-early-simplification=no-op-module -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-optimizer-early=no-op-module -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-optimizer-last=no-op-module -passes='default<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-full-link-time-optimization-early=no-op-module -passes='lto<O0>' 2>&1 < %s | FileCheck %s
; RUN: opt -disable-output -debug-pass-manager -passes-ep-full-link-time-optimization-last=no-op-module -passes='lto<O0>' 2>&1 < %s | FileCheck %s

; CHECK: Running pass: NoOp

declare void @bar() local_unnamed_addr

define void @foo(i32 %n) local_unnamed_addr {
entry:
  br label %loop
loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %loop ]
  %iv.next = add i32 %iv, 1
  tail call void @bar()
  %cmp = icmp eq i32 %iv, %n
  br i1 %cmp, label %exit, label %loop
exit:
  ret void
}
