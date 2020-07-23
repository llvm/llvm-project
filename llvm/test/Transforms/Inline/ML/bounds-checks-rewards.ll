; Test behavior when inlining policy grows size out of control.
; In all cases, the end result is the same: mandatory inlinings must happen.
; However, when we discover we 'trip' over the artificially-low size increase 
; factor, we penalize the 'bad' decision.
; REQUIRES: have_tf_api
; RUN: opt -passes=scc-oz-module-inliner -ml-inliner-ir2native-model=%S/../../../../unittests/Analysis/Inputs/ir2native_x86_64_model -ml-inliner-model-under-training=%S/../../../../lib/Analysis/models/inliner -training-log=- -enable-ml-inliner=development -ml-advisor-size-increase-threshold=10.0 -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=NOBOUNDS
; RUN: opt -passes=scc-oz-module-inliner -ml-inliner-ir2native-model=%S/../../../../unittests/Analysis/Inputs/ir2native_x86_64_model -ml-inliner-model-under-training=%S/../../../../lib/Analysis/models/inliner -training-log=- -enable-ml-inliner=development -ml-advisor-size-increase-threshold=1.0 -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=BOUNDS

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"

declare i64 @f1()

define i64 @f2() #0 {
  %r = call i64 @f1()
  %r2 = add i64 13, %r
  ret i64 %r2
}

define i64 @some_function() {
  %r = call i64 @f1()
  %r2 = add i64 13, %r
  ret i64 %r2
}

define i64 @top() {
  %r = call i64 @f2()
  %r2 = call i64 @some_function()
  %r3 = add i64 %r, %r2
  ret i64 %r3
}

attributes #0 = { alwaysinline }
; CHECK: key: "delta_size" value: {
; NOBOUNDS-NEXT: feature: { int64_list: { value: [10] } }
; NOBOUNDS-NEXT: feature: { int64_list: { value: [6] } }
; BOUNDS-NEXT: feature: { int64_list: { value: [2147483647] } }
; CHECK-NEXT: }
; CHECK-LABEL: @top
; f2 must always be inlined, so we won't find a call to it in @top()
; CHECK-NOT: call i64 @f2
; @some-function isn't mandatory, and when we set the increase threshold too low,
; it won't be inlined.
; NOBOUNDS-NOT: @some_function
; BOUNDS: call i64 @some_function