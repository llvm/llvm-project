; RUN: opt < %s -passes="print<inline-cost>" 2>&1 | FileCheck %s

; CHECK:       Analyzing call of foo... (caller:main)
; CHECK: define ptr @foo() {
; CHECK:  cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:  %1 = inttoptr i64 754974720 to ptr
; CHECK:  cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:  ret ptr %1
; CHECK: }
; CHECK:       NumConstantArgs: {{.*}}
; CHECK:       NumConstantOffsetPtrArgs: {{.*}}
; CHECK:       NumAllocaArgs: {{.*}}
; CHECK:       NumConstantPtrCmps: {{.*}}
; CHECK:       NumConstantPtrDiffs: {{.*}}
; CHECK:       NumInstructionsSimplified: {{.*}}
; CHECK:       NumInstructions: {{.*}}
; CHECK:       SROACostSavings: {{.*}}
; CHECK:       SROACostSavingsLost: {{.*}}
; CHECK:       LoadEliminationCost: {{.*}}
; CHECK:       ContainsNoDuplicateCall: {{.*}}
; CHECK:       Cost: {{.*}}
; CHECK:       Threshold: {{.*}}
; CHECK-EMPTY:
; CHECK:  Analyzing call of foo... (caller:main)

define ptr @foo() {
  %1 = inttoptr i64 754974720 to ptr
  ret ptr %1
}

define ptr @main() {
  %1 = call ptr @foo()
  %2 = call ptr @foo()
  ret ptr %1
}

; Make sure it also analyzes invoke call sites.

; CHECK:       Analyzing call of g... (caller:f)
; CHECK: define i32 @g(i32 %v) {
; CHECK: ; cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:   %p = icmp ugt i32 %v, 35
; CHECK: ; cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:   %r = select i1 %p, i32 %v, i32 7
; CHECK: ; cost before = {{.*}}, cost after = {{.*}}, threshold before = {{.*}}, threshold after = {{.*}}, cost delta = {{.*}}
; CHECK:   ret i32 %r
; CHECK: }
define i32 @g(i32 %v) {
  %p = icmp ugt i32 %v, 35
  %r = select i1 %p, i32 %v, i32 7
  ret i32 %r
}

define void @f(i32 %v, ptr %dst) personality ptr @__gxx_personality_v0 {
  %v1 = invoke i32 @g(i32 %v)
          to label %bb1 unwind label %bb2
bb1:
  store i32 %v1, ptr %dst
  ret void
bb2:
  %lpad.loopexit80 = landingpad { ptr, i32 }
          cleanup
  ret void
}

declare i32 @__gxx_personality_v0(...)
