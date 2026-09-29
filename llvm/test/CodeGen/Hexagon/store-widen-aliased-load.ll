; RUN: llc -mtriple=hexagon --combiner-store-merging=false -verify-machineinstrs < %s | FileCheck %s
; CHECK: memh
; Check that store widening merges the two adjacent stores.

target triple = "hexagon"

%struct.type_t = type { i8, i8, [2 x i8] }

define zeroext i8 @foo(ptr nocapture %p) nounwind {
entry:
  store i8 0, ptr %p, align 2
  %b = getelementptr inbounds %struct.type_t, ptr %p, i32 0, i32 1
  %0 = load i8, ptr %b, align 1
  store i8 0, ptr %b, align 1
  ret i8 %0
}
