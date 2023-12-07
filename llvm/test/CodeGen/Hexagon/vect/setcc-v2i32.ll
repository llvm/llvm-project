; RUN: llc -march=hexagon < %s | FileCheck %s
; REQUIRES: asserts

; Check that this testcase doesn't crash.
; CHECK: vcmpw.eq

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define i32 @fred(ptr %a0) #0 {
b1:
  %v2 = load <2 x i16>, ptr %a0, align 2
  %v3 = icmp eq <2 x i16> %v2, zeroinitializer
  %v4 = zext <2 x i1> %v3 to <2 x i16>
  %v5 = extractelement <2 x i16> %v4, i32 1
  %v8 = icmp ne i16 %v5, 1
  %v9 = zext i1 %v8 to i32
  ret i32 %v9
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
