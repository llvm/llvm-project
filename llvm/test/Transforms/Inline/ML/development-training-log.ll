; Test that we can produce a log if we have or do not have a model, in development mode.
; REQUIRES: have_tf_api
; RUN: opt -enable-ml-inliner=development -passes=scc-oz-module-inliner -training-log=- -ml-inliner-model-under-training=%S/../../../../lib/Analysis/models/inliner -ml-inliner-ir2native-model=%S/../../../../unittests/Analysis/Inputs/ir2native_x86_64_model -S < %s | FileCheck %s 
; RUN: opt -enable-ml-inliner=development -passes=scc-oz-module-inliner -training-log=- -ml-inliner-ir2native-model=%S/../../../../unittests/Analysis/Inputs/ir2native_x86_64_model -S < %s | FileCheck %s

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

; Check we produce a protobuf that has inlining decisions and rewards.
; CHECK:     feature_lists: {
; CHECK:          key: "inlining_decision" value: {
; CHECK-NEXT:       feature: { int64_list: { value: [1] } }
; CHECK-NEXT:     }
; CHECK-NEXT:   }
; CHECK-NEXT:   feature_list: {
; CHECK-NEXT:     key: "delta_size" value: {
; CHECK-NEXT:       feature: { int64_list: { value: [0] } }
; CHECK-NEXT:     }
; CHECK-NEXT:   }