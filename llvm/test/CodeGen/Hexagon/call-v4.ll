; RUN: llc -mtriple=hexagon -print-after=finalize-isel -o /dev/null 2>&1 < %s | FileCheck %s
; REQUIRES: asserts

; CHECK: J2_call @f1
; CHECK: PS_call_nr @f2

target triple = "hexagon"

@g0 = external global i32

; Function Attrs: nounwind
define i32 @f0() #0 {
b0:
  %v0 = load i32, ptr @g0, align 4
  %v1 = tail call i32 @f1(i32 %v0) #0
  %v2 = icmp eq i32 %v1, 0
  br i1 %v2, label %b1, label %b2

b1:                                               ; preds = %b0
  tail call void @f2() #2
  unreachable

b2:                                               ; preds = %b0
  ret i32 0
}

declare i32 @f1(i32)

; Function Attrs: noreturn
declare void @f2() #1

attributes #0 = { nounwind "disable-tail-calls"="true" }
attributes #1 = { noreturn }
attributes #2 = { noreturn nounwind }
