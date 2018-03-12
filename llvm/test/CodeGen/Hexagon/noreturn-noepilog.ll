; RUN: llc -march=hexagon < %s | FileCheck %s
; Check that no epilogue is inserted after a noreturn call.
;
; CHECK-LABEL: f1:
; CHECK: allocframe(r29,#0):raw
; CHECK-NOT: deallocframe

target triple = "hexagon"

%s.0 = type <{ i16, i8, i8, i8 }>

@g0 = internal constant %s.0 <{ i16 1, i8 2, i8 3, i8 4 }>, align 4

; Function Attrs: noreturn
declare void @f0(%s.0*, i32) #0

define i64 @f1(i32 %a0, i32 %a1) {
b0:
  %v0 = icmp ugt i32 %a0, 3
  br i1 %v0, label %b1, label %b2

b1:                                               ; preds = %b0
  call void @f0(%s.0* nonnull @g0, i32 %a0) #0
  unreachable

b2:                                               ; preds = %b0
  %v1 = mul i32 %a1, 7
  %v2 = zext i32 %v1 to i64
  ret i64 %v2
}

attributes #0 = { noreturn }
