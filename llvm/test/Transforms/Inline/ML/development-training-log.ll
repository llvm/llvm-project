; Test that we can produce a log if we have or do not have a model, in development mode.
; REQUIRES: have_tflite
; Generate mock model
; RUN: rm -rf %t_savedmodel %t
; RUN: %python %S/../../../../lib/Analysis/models/gen-inline-oz-test-model.py %t_savedmodel
; RUN: %python %S/../../../../lib/Analysis/models/saved-model-to-tflite.py %t_savedmodel %t
;
; RUN: opt -enable-ml-inliner=development -passes=scc-oz-module-inliner -training-log=%t4 -ml-inliner-model-under-training=%t -S < %s
; RUN: %python %S/../../../../lib/Analysis/models/log_reader.py %t4 | FileCheck %s --check-prefix=NOREWARD
; RUN: opt -enable-ml-inliner=development -passes=scc-oz-module-inliner -training-log=%t5 -S < %s
; RUN: %python %S/../../../../lib/Analysis/models/log_reader.py %t5| FileCheck %s --check-prefix=NOREWARD
target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"
declare i32 @f1(i32)
declare i32 @f2(i32)
define dso_local i32 @branches(i32) {
  %cond = icmp slt i32 %0, 3
  br i1 %cond, label %then, label %else
then:
  %ret.1 = call i32 @f1(i32 %0)
  br label %last.block
else:
  %ret.2 = call i32 @f2(i32 %0)
  br label %last.block
last.block:
  %ret = phi i32 [%ret.1, %then], [%ret.2, %else]
  ret i32 %ret
}
define dso_local i32 @top() {
  %1 = call i32 @branches(i32 2)
  ret i32 %1
}
!llvm.module.flags = !{!0}
!llvm.ident = !{!1}
!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 7.0.0-6 (tags/RELEASE_700/final)"}
; Check we produce a log that has inlining decisions and rewards.
; CHECK-NOT:        fake_extra_output:
; EXTRA-OUTPUTS:    fake_extra_output: {{[0-9]+}}
; CHECK:            inlining_decision: 1
; CHECK:            delta_size: 0
; NOREWARD-NOT:     delta_size:
