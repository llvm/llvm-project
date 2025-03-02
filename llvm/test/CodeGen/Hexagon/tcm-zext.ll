; RUN: llc -mtriple=hexagon < %s | FileCheck %s
; CHECK: jump f2
;
; Check that we perform tail call merging on return types with zero extend.
; We want to see a jump to f2, not a call.

target triple = "hexagon"

; Function Attrs: nounwind
define zeroext i8 @f0() #0 {
b0:
  %v0 = tail call zeroext i8 @f2() #0
  ret i8 %v0
}

; Function Attrs: nounwind readnone
define zeroext i8 @f1() #1 {
b0:
  ret i8 1
}

declare zeroext i8 @f2()

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
