; RUN: llc -march=hexagon < %s | FileCheck %s

; Test that the first packet contains 4 instructions, which includes
; a call. The compiler marked the call as constant extended incorrectly,
; which meant it couldn't fit in the first packet. But, calls are not
; constant extended by the compiler.

; CHECK: {
; CHECK-NEXT: call f1
; CHECK-NEXT: combine
; CHECK-NEXT: memd
; CHECK-NEXT: allocframe
; CHECK-NEXT: }


@g0 = external global i32

; Function Attrs: noinline nounwind
define i32 @f0(i32 %a0, i32* nocapture %a1) #0 {
b0:
  %v0 = tail call i32 @f1(i32 %a0)
  %v1 = icmp eq i32 %v0, 0
  %v2 = select i1 %v1, i32 3, i32 %a0
  store i32 %v2, i32* %a1, align 4
  switch i32 %a0, label %b5 [
    i32 0, label %b1
    i32 1, label %b2
    i32 2, label %b3
    i32 4, label %b4
  ]

b1:                                               ; preds = %b0
  store i32 0, i32* %a1, align 4
  br label %b5

b2:                                               ; preds = %b0
  %v3 = load i32, i32* @g0, align 4
  %v4 = icmp sgt i32 %v3, 100
  %v5 = select i1 %v4, i32 0, i32 3
  store i32 %v5, i32* %a1, align 4
  br label %b5

b3:                                               ; preds = %b0
  store i32 1, i32* %a1, align 4
  br label %b5

b4:                                               ; preds = %b0
  store i32 2, i32* %a1, align 4
  br label %b5

b5:                                               ; preds = %b4, %b3, %b2, %b1, %b0
  ret i32 undef
}

; Function Attrs: noinline nounwind readnone
declare i32 @f1(i32) #1

attributes #0 = { noinline nounwind "target-cpu"="hexagonv60" }
attributes #1 = { noinline nounwind readnone "target-cpu"="hexagonv60" }
