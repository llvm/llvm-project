; RUN: llc -march=hexagon -verify-machineinstrs < %s | FileCheck %s
;
; Make sure this test compiles successfully.
; CHECK: call foo

target triple = "hexagon--elf"

; Function Attrs: nounwind
define i32 @fred(i1 %a0) #0 {
b0:
  call void @foo() #0
  br label %b1

b1:                                               ; preds = %b0
  br i1 %a0, label %b2, label %b3

b2:                                               ; preds = %b1
  br label %b3

b3:                                               ; preds = %b2, %b1
  %v4 = phi i32 [ 1, %b1 ], [ 2, %b2 ]
  ret i32 %v4
}

declare void @foo() #0

attributes #0 = { nounwind "target-cpu"="hexagonv60" }
