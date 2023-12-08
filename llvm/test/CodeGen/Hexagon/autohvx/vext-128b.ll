; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00:
; CHECK: v1:0.h = vunpack(v0.b)
define <128 x i16> @test_00(<128 x i8> %v0) #0 {
  %p = sext <128 x i8> %v0 to <128 x i16>
  ret <128 x i16> %p
}

; CHECK-LABEL: test_01:
; CHECK: v1:0.w = vunpack(v0.h)
define <64 x i32> @test_01(<64 x i16> %v0) #0 {
  %p = sext <64 x i16> %v0 to <64 x i32>
  ret <64 x i32> %p
}

; CHECK-LABEL: test_02:
; CHECK: v1:0.uh = vunpack(v0.ub)
define <128 x i16> @test_02(<128 x i8> %v0) #0 {
  %p = zext <128 x i8> %v0 to <128 x i16>
  ret <128 x i16> %p
}

; CHECK-LABEL: test_03:
; CHECK: v1:0.uw = vunpack(v0.uh)
define <64 x i32> @test_03(<64 x i16> %v0) #0 {
  %p = zext <64 x i16> %v0 to <64 x i32>
  ret <64 x i32> %p
}

; CHECK-LABEL: test_04:
; CHECK: v[[H40:[0-9]+]]:[[L40:[0-9]+]].h = vunpack(v0.b)
; CHECK: v1:0.w = vunpack(v[[L40]].h)
define <32 x i32> @test_04(<128 x i8> %v0) #0 {
  %x = sext <128 x i8> %v0 to <128 x i32>
  %p = shufflevector <128 x i32> %x, <128 x i32> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <32 x i32> %p
}

; CHECK-LABEL: test_05:
; CHECK: v[[H50:[0-9]+]]:[[L50:[0-9]+]].uh = vunpack(v0.ub)
; CHECK: v1:0.uw = vunpack(v[[L50]].uh)
define <32 x i32> @test_05(<128 x i8> %v0) #0 {
  %x = zext <128 x i8> %v0 to <128 x i32>
  %p = shufflevector <128 x i32> %x, <128 x i32> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <32 x i32> %p
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }

