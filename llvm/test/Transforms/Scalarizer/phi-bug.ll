; RUN: opt %s -passes='function(scalarizer,verify)' -S -o - | FileCheck %s

define void @f3() local_unnamed_addr {
bb1:
  br label %bb2

bb3:
; CHECK-LABEL: bb3:
; CHECK-NEXT: br label %bb4
  %h.10.0.vec.insert = shufflevector <1 x i16> %h.10.1, <1 x i16> undef, <1 x i32> <i32 0>
  br label %bb4

bb2:
; CHECK-LABEL: bb2:
; CHECK: phi i16
  %h.10.1 = phi <1 x i16> [ undef, %bb1 ]
  br label %bb3

bb4:
; CHECK-LABEL: bb4:
; CHECK: phi i16
  %h.10.2 = phi <1 x i16> [ %h.10.0.vec.insert, %bb3 ]
  ret void
}
