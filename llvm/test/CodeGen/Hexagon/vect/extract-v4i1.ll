; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; Test that the compiler generates the correct code when sign-extending a
; predicate register when it is converted from one vector predicate type
; to another. In this case, the compiler generates two v4i1 EXTRACT_SUBVECT
; from a v8i1, for the lower and upper parts.

; CHECK: r[[REGH:([0-9]+)]]:[[REGL:([0-9]+)]] = mask(p{{[0-3]}})
; CHECK-DAG: = vsxtbh(r[[REGH]])
; CHECK-DAG: = vsxtbh(r[[REGL]])

target triple = "hexagon"

define void @f0(ptr %a0, ptr %a1) #0 {
b0:
  %v0 = load i16, ptr %a0, align 2
  %v1 = sext i16 %v0 to i32
  %v2 = insertelement <8 x i32> undef, i32 %v1, i32 0
  %v3 = shufflevector <8 x i32> %v2, <8 x i32> undef, <8 x i32> zeroinitializer
  %v4 = shl <8 x i32> <i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1, i32 1>, <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  %v5 = and <8 x i32> %v4, %v3
  %v6 = icmp ne <8 x i32> %v5, zeroinitializer
  %v7 = zext <8 x i1> %v6 to <8 x i16>
  store <8 x i16> %v7, ptr %a1, align 8
  ret void
}

attributes #0 = { nounwind optsize "target-cpu"="hexagonv65" "target-features"="+hvx-length64b,+hvxv65,-long-calls" }
