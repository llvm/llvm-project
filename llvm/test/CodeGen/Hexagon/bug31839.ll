; RUN: llc -mtriple=hexagon < %s
; REQUIRES: asserts

; Check for successful compilation.

define ptr @f0(i32 %a0, i32 %a1) {
b0:
  %v0 = call noalias ptr @f1(i32 undef, i32 undef)
  br i1 undef, label %b2, label %b1

b1:                                               ; preds = %b0
  %v1 = ptrtoint ptr %v0 to i32
  store volatile i32 %v1, ptr %v0, align 4
  %v3 = getelementptr inbounds i8, ptr %v0, i32 4
  store ptr %v0, ptr %v3, align 4
  %v5 = getelementptr inbounds i8, ptr %v0, i32 16
  br label %b2

b2:                                               ; preds = %b1, %b0
  %v6 = phi ptr [ %v5, %b1 ], [ null, %b0 ]
  ret ptr %v6
}

declare noalias ptr @f1(i32, i32) local_unnamed_addr
