; Test behavior when inlining policy grows size out of control.
; In all cases, the end result is the same: mandatory inlinings must happen.
; Also in all cases, we don't record the mandatory inlining (there's nothing to
; learn from it).
; However, when we discover we 'trip' over the artificially-low size increase 
; factor, we penalize the 'bad' decision.
; REQUIRES: have_tf_api
;
; Generate mock model
; RUN: rm -rf %t
; RUN: %python %S/../../../../lib/Analysis/models/gen-inline-oz-test-model.py %t
;
; When the bounds are very wide ("no bounds"), all inlinings happen.
; RUN: opt -passes=scc-oz-module-inliner -ml-inliner-ir2native-model=%S/../../../../unittests/Analysis/Inputs/ir2native_x86_64_model -ml-inliner-model-under-training=%t -training-log=- -tfutils-text-log -enable-ml-inliner=development -ml-advisor-size-increase-threshold=10.0 -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=NOBOUNDS
;
; When the bounds are very restrictive, the first inlining happens but it's
; considered as "bad" (since it trips over the bounds) and its reward is a
; penalty. However, the mandatory inlining, which is considered next, happens.
; No other inlinings happend.
; RUN: opt -passes=scc-oz-module-inliner -ml-inliner-ir2native-model=%S/../../../../unittests/Analysis/Inputs/ir2native_x86_64_model -ml-inliner-model-under-training=%t -training-log=- -tfutils-text-log -enable-ml-inliner=development -ml-advisor-size-increase-threshold=1.0 -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=BOUNDS
;
; With more restrictive bounds, the first inlining happens and is OK. The
; mandatory inlining happens next, and it trips over the bounds, which then
; forces no further inlinings.
; RUN: opt -passes=scc-oz-module-inliner -ml-inliner-ir2native-model=%S/../../../../unittests/Analysis/Inputs/ir2native_x86_64_model -ml-inliner-model-under-training=%t -training-log=- -tfutils-text-log -enable-ml-inliner=development -ml-advisor-size-increase-threshold=1.1 -S < %s 2>&1 | FileCheck %s --check-prefix=CHECK --check-prefix=RELAXED-BOUNDS
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-grtev4-linux-gnu"
declare i64 @f1()
define i64 @may_not_be_inlined() {
  %r = call i64 @f1()
  %r2 = add i64 13, %r
  ret i64 %r2
}
define i64 @must_be_inlined() #0 {
  %r = call i64 @may_not_be_inlined()
  %r2 = add i64 13, %r
  ret i64 %r2
}
define i64 @top() {
  %r = call i64 @must_be_inlined()
  %r2 = call i64 @may_not_be_inlined()
  %r3 = call i64 @may_not_be_inlined()
  %r4 = add i64 %r, %r2
  %r5 = add i64 %r3, %r4
  ret i64 %r5
}
attributes #0 = { alwaysinline }
; CHECK:       key: "delta_size"
; CHECK-NEXT:     value {
; CHECK-NEXT:       feature {
; CHECK-NEXT:         int64_list {
; NOBOUNDS-NEXT:        value: 6
; RELAXED-BOUNDS-NEXT:  value: 6
; NOBOUNDS-NEXT:      }
; NOBOUNDS-NEXT:    }
; NOBOUNDS-NEXT:    feature {
; NOBOUNDS-NEXT:      int64_list {
; NOBOUNDS-NEXT:        value: -11
; NOBOUNDS-NEXT:      }
; NOBOUNDS-NEXT:    }
; NOBOUNDS-NEXT:    feature {
; NOBOUNDS-NEXT:      int64_list {
; NOBOUNDS-NEXT:        value: 4
; BOUNDS-NEXT:          value: 2147483647
; CHECK-NEXT:         }
; CHECK-LABEL: @top
; must_be_inlined must always be inlined, so we won't find a call to it in @top()
; CHECK-NOT: call i64 @must_be_inlined
; @some-function isn't mandatory, and when we set the increase threshold too low,
; it won't be inlined.
; NOBOUNDS-NOT: @may_not_be_inlined
; RELAXED-BOUNDS: call i64 @may_not_be_inlined
; BOUNDS: call i64 @may_not_be_inlined
