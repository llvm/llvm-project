; RUN: llc -mtriple=hexagon -spill-func-threshold-Os=0 -spill-func-threshold=0 < %s | FileCheck %s

; Check that the long-calls feature handles save and restore.
; This is a test commit.
; CHECK: call ##__save
; CHECK: jump ##__restore

target triple = "hexagon"

; Function Attrs: nounwind
define i32 @f0(i32 %a0, i32 %a1, i32 %a2) #0 {
b0:
  %v0 = tail call i32 @f1(i32 %a0, i32 %a1, i32 %a2) #1
  %v1 = tail call i32 @f2(i32 %a0, i32 %a1, i32 %a2) #1
  ret i32 0
}

; Function Attrs: nounwind
declare i32 @f1(...) #1

; Function Attrs: nounwind
declare i32 @f2(...) #1

attributes #0 = { nounwind "target-features"="+long-calls" }
attributes #1 = { nounwind }
