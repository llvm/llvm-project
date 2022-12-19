; RUN: llc -march=hexagon < %s | FileCheck %s
;
; Check that we are emitting a regular call instead of a tail call for a
; noreturn call in a function with a non-empty frame (to save instructions).
;
; CHECK: call f0
; CHECK-NOT: deallocframe

target triple = "hexagon"

; Function Attrs: noreturn
declare void @f0(i32, ptr) #0

declare void @f1(ptr)

define i64 @f2(i32 %a0, i32 %a1) {
b0:
  %v0 = alloca i32
  call void @f1(ptr %v0)
  %v1 = icmp ugt i32 %a0, 3
  br i1 %v1, label %b1, label %b2

b1:                                               ; preds = %b0
  tail call void @f0(i32 %a0, ptr %v0) #0
  unreachable

b2:                                               ; preds = %b0
  %v2 = mul i32 %a1, 7
  %v3 = zext i32 %v2 to i64
  ret i64 %v3
}

attributes #0 = { noreturn }
