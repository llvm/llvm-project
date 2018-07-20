; RUN: llc -march=hexagon -hexagon-small-data-threshold=0 -disable-hexagon-amodeopt -hexagon-cext-threshold=1 < %s | FileCheck %s
; Check commoning of global addresses.

@g0 = external global i32

; Function Attrs: nounwind
define zeroext i32 @f0() #0 {
b0:
; CHECK: ##g0
; CHECK-NOT: ##g0
  %v0 = load i32, i32* @g0, align 1
  %v1 = mul nsw i32 100, %v0
  store i32 %v1, i32* @g0, align 1
  ret i32 %v1
}

attributes #0 = { nounwind }
