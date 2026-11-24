; RUN: opt -mtriple=nvptx64-nvidia-cuda -passes=load-store-vectorizer -S < %s | FileCheck %s

; The element offsets here sit near INT64_MAX, so computing `offset +
; storeSize` in the index width wraps around.  A wrapped read-end compares as
; less than its predecessor, which used to drop the i64 from the chain's byte
; extent and leave ChainBytes == 1 for a chain containing an 8-byte element.
; vectorizeChain then produced a single-lane b8 "vector" and tried to cast it
; straight to i64, asserting with "Invalid cast!".  The extent must be computed
; in a widened APInt instead.

define i32 @chain_extent_overflow(ptr %x) {
; CHECK-LABEL: define i32 @chain_extent_overflow(
; CHECK:         ret i32 0
entry:
  %G1 = getelementptr i1, ptr %x, i64 9223372036854775807
  %L5 = load i64, ptr %G1, align 4
  %G3 = getelementptr double, ptr %x, i1 true
  %L9 = load i8, ptr %G3, align 1
  %L6 = load i64, ptr %G3, align 4
  ret i32 0
}
