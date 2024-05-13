; RUN: llc -march=hexagon < %s | FileCheck %s

; Check that this compiles successfully.
; CHECK: vsplat

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

define void @fred(ptr %a0) #0 {
b0:
  %v1 = load <64 x i8>, ptr %a0, align 8
  %v2 = zext <64 x i8> %v1 to <64 x i32>
  %v3 = add nuw nsw <64 x i32> %v2, zeroinitializer
  %v4 = icmp ugt <64 x i32> %v3, <i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254, i32 254>
  %v5 = zext <64 x i1> %v4 to <64 x i32>
  %v6 = add nuw nsw <64 x i32> %v3, %v5
  %v7 = trunc <64 x i32> %v6 to <64 x i8>
  store <64 x i8> %v7, ptr %a0, align 8
  ret void
}

attributes #0 = { nounwind "target-cpu"="hexagonv60" "target-features"="+hvx-length64b,+hvxv60" }
