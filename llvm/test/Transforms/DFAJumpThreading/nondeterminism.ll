; RUN: opt -passes=dfa-jump-threading -S --preserve-ll-uselistorder < %s | FileCheck %s

; Check that threading produces deterministic use lists for %v and %w.

define void @f(i64 %v, i64 %w) {
; CHECK-LABEL: define void @f(
; CHECK: uselistorder i64 %v, { 2, 0, 4, 3, 1, 5 }
; CHECK: uselistorder i64 %w, { 2, 0, 4, 3, 1, 5 }
entry:
  br label %header

header:
  br i1 false, label %join, label %side

side:
  br label %join

join:
  %state = phi i32 [ 0, %side ], [ 1, %header ]
  br label %p1

p1:
  br label %use1

use1:
  %phi1 = phi i64 [ %v, %p1 ]
  br label %p2

p2:
  br label %use2

use2:
  %phi2 = phi i64 [ %w, %p2 ]
  br label %p3

p3:
  br label %use3

use3:
  %phi3 = phi i64 [ %v, %p3 ]
  br label %p4

p4:
  br label %use4

use4:
  %phi4 = phi i64 [ %w, %p4 ]
  br label %latch

latch:
  switch i32 %state, label %exit [
    i32 0, label %header
    i32 2, label %header
  ]

exit:
  ret void
}
