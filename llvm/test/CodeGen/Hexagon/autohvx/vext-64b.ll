; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00:
; CHECK: v1:0.h = vunpack(v0.b)
define <64 x i16> @test_00(<64 x i8> %v0) #0 {
  %p = sext <64 x i8> %v0 to <64 x i16>
  ret <64 x i16> %p
}

; CHECK-LABEL: test_01:
; CHECK: v1:0.w = vunpack(v0.h)
define <32 x i32> @test_01(<32 x i16> %v0) #0 {
  %p = sext <32 x i16> %v0 to <32 x i32>
  ret <32 x i32> %p
}

; CHECK-LABEL: test_02:
; CHECK: v1:0.uh = vunpack(v0.ub)
define <64 x i16> @test_02(<64 x i8> %v0) #0 {
  %p = zext <64 x i8> %v0 to <64 x i16>
  ret <64 x i16> %p
}

; CHECK-LABEL: test_03:
; CHECK: v1:0.uw = vunpack(v0.uh)
define <32 x i32> @test_03(<32 x i16> %v0) #0 {
  %p = zext <32 x i16> %v0 to <32 x i32>
  ret <32 x i32> %p
}

; CHECK-LABEL: test_04:
; CHECK-DAG: v[[H40:[0-9]+]]:[[L40:[0-9]+]].h = vunpack(v0.b)
; CHECK: v1:0.w = vunpack(v[[L40]].h)
define <16 x i32> @test_04(<64 x i8> %v0) #0 {
  %x = sext <64 x i8> %v0 to <64 x i32>
  %p = shufflevector <64 x i32> %x, <64 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i32> %p
}

; CHECK-LABEL: test_05:
; CHECK-DAG: v[[H50:[0-9]+]]:[[L50:[0-9]+]].uh = vunpack(v0.ub)
; CHECK: v1:0.uw = vunpack(v[[L50]].uh)
define <16 x i32> @test_05(<64 x i8> %v0) #0 {
  %x = zext <64 x i8> %v0 to <64 x i32>
  %p = shufflevector <64 x i32> %x, <64 x i32> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  ret <16 x i32> %p
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

