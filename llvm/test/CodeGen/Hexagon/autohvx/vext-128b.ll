; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test_00:
; CHECK-DAG: v[[H00:[0-9]+]]:[[L00:[0-9]+]].h = vsxt(v0.b)
; CHECK-DAG: r[[R00:[0-9]+]] = #-2
; CHECK: v1:0 = vshuff(v[[H00]],v[[L00]],r[[R00]])
define <128 x i16> @test_00(<128 x i8> %v0) #0 {
  %p = sext <128 x i8> %v0 to <128 x i16>
  ret <128 x i16> %p
}

; CHECK-LABEL: test_01:
; CHECK-DAG: v[[H10:[0-9]+]]:[[L10:[0-9]+]].w = vsxt(v0.h)
; CHECK-DAG: r[[R10:[0-9]+]] = #-4
; CHECK: v1:0 = vshuff(v[[H10]],v[[L10]],r[[R10]])
define <64 x i32> @test_01(<64 x i16> %v0) #0 {
  %p = sext <64 x i16> %v0 to <64 x i32>
  ret <64 x i32> %p
}

; CHECK-LABEL: test_02:
; CHECK-DAG: v[[H20:[0-9]+]]:[[L20:[0-9]+]].uh = vzxt(v0.ub)
; CHECK-DAG: r[[R20:[0-9]+]] = #-2
; CHECK: v1:0 = vshuff(v[[H20]],v[[L20]],r[[R20]])
define <128 x i16> @test_02(<128 x i8> %v0) #0 {
  %p = zext <128 x i8> %v0 to <128 x i16>
  ret <128 x i16> %p
}

; CHECK-LABEL: test_03:
; CHECK-DAG: v[[H30:[0-9]+]]:[[L30:[0-9]+]].uw = vzxt(v0.uh)
; CHECK-DAG: r[[R30:[0-9]+]] = #-4
; CHECK: v1:0 = vshuff(v[[H30]],v[[L30]],r[[R30]])
define <64 x i32> @test_03(<64 x i16> %v0) #0 {
  %p = zext <64 x i16> %v0 to <64 x i32>
  ret <64 x i32> %p
}

; CHECK-LABEL: test_04:
; CHECK-DAG: v[[H40:[0-9]+]]:[[L40:[0-9]+]].h = vsxt(v0.b)
; CHECK-DAG: r[[R40:[0-9]+]] = #-2
; CHECK-DAG: r[[R41:[0-9]+]] = #-4
; CHECK: v[[H41:[0-9]+]]:[[L41:[0-9]+]] = vshuff(v[[H40]],v[[L40]],r[[R40]])
; CHECK: v[[H42:[0-9]+]]:[[L42:[0-9]+]].w = vsxt(v[[L41]].h)
; CHECK: v1:0 = vshuff(v[[H42]],v[[L42]],r[[R41]])
define <32 x i32> @test_04(<128 x i8> %v0) #0 {
  %x = sext <128 x i8> %v0 to <128 x i32>
  %p = shufflevector <128 x i32> %x, <128 x i32> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <32 x i32> %p
}

; CHECK-LABEL: test_05:
; CHECK-DAG: v[[H50:[0-9]+]]:[[L50:[0-9]+]].uh = vzxt(v0.ub)
; CHECK-DAG: r[[R50:[0-9]+]] = #-2
; CHECK-DAG: r[[R51:[0-9]+]] = #-4
; CHECK: v[[H51:[0-9]+]]:[[L51:[0-9]+]] = vshuff(v[[H50]],v[[L50]],r[[R50]])
; CHECK: v[[H52:[0-9]+]]:[[L52:[0-9]+]].uw = vzxt(v[[L51]].uh)
; CHECK: v1:0 = vshuff(v[[H52]],v[[L52]],r[[R51]])
define <32 x i32> @test_05(<128 x i8> %v0) #0 {
  %x = zext <128 x i8> %v0 to <128 x i32>
  %p = shufflevector <128 x i32> %x, <128 x i32> undef, <32 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 16, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31>
  ret <32 x i32> %p
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }

