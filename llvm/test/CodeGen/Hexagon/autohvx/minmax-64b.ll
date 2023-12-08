; RUN: llc -march=hexagon < %s | FileCheck %s

; minb

; CHECK: test_00:
; CHECK: v0.b = vmin(v0.b,v1.b)
define <64 x i8> @test_00(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp slt <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v0, <64 x i8> %v1
  ret <64 x i8> %t1
}

; CHECK: test_01:
; CHECK: v0.b = vmin(v0.b,v1.b)
define <64 x i8> @test_01(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp sle <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v0, <64 x i8> %v1
  ret <64 x i8> %t1
}

; CHECK: test_02:
; CHECK: v0.b = vmin(v0.b,v1.b)
define <64 x i8> @test_02(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp sgt <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v1, <64 x i8> %v0
  ret <64 x i8> %t1
}

; CHECK: test_03:
; CHECK: v0.b = vmin(v0.b,v1.b)
define <64 x i8> @test_03(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp sge <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v1, <64 x i8> %v0
  ret <64 x i8> %t1
}

; maxb

; CHECK: test_04:
; CHECK: v0.b = vmax(v0.b,v1.b)
define <64 x i8> @test_04(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp slt <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v1, <64 x i8> %v0
  ret <64 x i8> %t1
}

; CHECK: test_05:
; CHECK: v0.b = vmax(v0.b,v1.b)
define <64 x i8> @test_05(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp sle <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v1, <64 x i8> %v0
  ret <64 x i8> %t1
}

; CHECK: test_06:
; CHECK: v0.b = vmax(v0.b,v1.b)
define <64 x i8> @test_06(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp sgt <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v0, <64 x i8> %v1
  ret <64 x i8> %t1
}

; CHECK: test_07:
; CHECK: v0.b = vmax(v0.b,v1.b)
define <64 x i8> @test_07(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp sge <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v0, <64 x i8> %v1
  ret <64 x i8> %t1
}

; minub

; CHECK: test_08:
; CHECK: v0.ub = vmin(v0.ub,v1.ub)
define <64 x i8> @test_08(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp ult <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v0, <64 x i8> %v1
  ret <64 x i8> %t1
}

; CHECK: test_09:
; CHECK: v0.ub = vmin(v0.ub,v1.ub)
define <64 x i8> @test_09(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp ule <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v0, <64 x i8> %v1
  ret <64 x i8> %t1
}

; CHECK: test_0a:
; CHECK: v0.ub = vmin(v0.ub,v1.ub)
define <64 x i8> @test_0a(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp ugt <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v1, <64 x i8> %v0
  ret <64 x i8> %t1
}

; CHECK: test_0b:
; CHECK: v0.ub = vmin(v0.ub,v1.ub)
define <64 x i8> @test_0b(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp uge <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v1, <64 x i8> %v0
  ret <64 x i8> %t1
}

; maxub

; CHECK: test_0c:
; CHECK: v0.ub = vmax(v0.ub,v1.ub)
define <64 x i8> @test_0c(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp ult <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v1, <64 x i8> %v0
  ret <64 x i8> %t1
}

; CHECK: test_0d:
; CHECK: v0.ub = vmax(v0.ub,v1.ub)
define <64 x i8> @test_0d(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp ule <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v1, <64 x i8> %v0
  ret <64 x i8> %t1
}

; CHECK: test_0e:
; CHECK: v0.ub = vmax(v0.ub,v1.ub)
define <64 x i8> @test_0e(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp ugt <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v0, <64 x i8> %v1
  ret <64 x i8> %t1
}

; CHECK: test_0f:
; CHECK: v0.ub = vmax(v0.ub,v1.ub)
define <64 x i8> @test_0f(<64 x i8> %v0, <64 x i8> %v1) #0 {
  %t0 = icmp uge <64 x i8> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i8> %v0, <64 x i8> %v1
  ret <64 x i8> %t1
}

; minh

; CHECK: test_10:
; CHECK: v0.h = vmin(v0.h,v1.h)
define <32 x i16> @test_10(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp slt <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v0, <32 x i16> %v1
  ret <32 x i16> %t1
}

; CHECK: test_11:
; CHECK: v0.h = vmin(v0.h,v1.h)
define <32 x i16> @test_11(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp sle <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v0, <32 x i16> %v1
  ret <32 x i16> %t1
}

; CHECK: test_12:
; CHECK: v0.h = vmin(v0.h,v1.h)
define <32 x i16> @test_12(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp sgt <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v1, <32 x i16> %v0
  ret <32 x i16> %t1
}

; CHECK: test_13:
; CHECK: v0.h = vmin(v0.h,v1.h)
define <32 x i16> @test_13(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp sge <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v1, <32 x i16> %v0
  ret <32 x i16> %t1
}

; maxh

; CHECK: test_14:
; CHECK: v0.h = vmax(v0.h,v1.h)
define <32 x i16> @test_14(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp slt <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v1, <32 x i16> %v0
  ret <32 x i16> %t1
}

; CHECK: test_15:
; CHECK: v0.h = vmax(v0.h,v1.h)
define <32 x i16> @test_15(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp sle <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v1, <32 x i16> %v0
  ret <32 x i16> %t1
}

; CHECK: test_16:
; CHECK: v0.h = vmax(v0.h,v1.h)
define <32 x i16> @test_16(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp sgt <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v0, <32 x i16> %v1
  ret <32 x i16> %t1
}

; CHECK: test_17:
; CHECK: v0.h = vmax(v0.h,v1.h)
define <32 x i16> @test_17(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp sge <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v0, <32 x i16> %v1
  ret <32 x i16> %t1
}

; minuh

; CHECK: test_18:
; CHECK: v0.uh = vmin(v0.uh,v1.uh)
define <32 x i16> @test_18(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp ult <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v0, <32 x i16> %v1
  ret <32 x i16> %t1
}

; CHECK: test_19:
; CHECK: v0.uh = vmin(v0.uh,v1.uh)
define <32 x i16> @test_19(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp ule <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v0, <32 x i16> %v1
  ret <32 x i16> %t1
}

; CHECK: test_1a:
; CHECK: v0.uh = vmin(v0.uh,v1.uh)
define <32 x i16> @test_1a(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp ugt <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v1, <32 x i16> %v0
  ret <32 x i16> %t1
}

; CHECK: test_1b:
; CHECK: v0.uh = vmin(v0.uh,v1.uh)
define <32 x i16> @test_1b(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp uge <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v1, <32 x i16> %v0
  ret <32 x i16> %t1
}

; maxuh

; CHECK: test_1c:
; CHECK: v0.uh = vmax(v0.uh,v1.uh)
define <32 x i16> @test_1c(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp ult <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v1, <32 x i16> %v0
  ret <32 x i16> %t1
}

; CHECK: test_1d:
; CHECK: v0.uh = vmax(v0.uh,v1.uh)
define <32 x i16> @test_1d(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp ule <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v1, <32 x i16> %v0
  ret <32 x i16> %t1
}

; CHECK: test_1e:
; CHECK: v0.uh = vmax(v0.uh,v1.uh)
define <32 x i16> @test_1e(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp ugt <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v0, <32 x i16> %v1
  ret <32 x i16> %t1
}

; CHECK: test_1f:
; CHECK: v0.uh = vmax(v0.uh,v1.uh)
define <32 x i16> @test_1f(<32 x i16> %v0, <32 x i16> %v1) #0 {
  %t0 = icmp uge <32 x i16> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i16> %v0, <32 x i16> %v1
  ret <32 x i16> %t1
}

; minw

; CHECK: test_20:
; CHECK: v0.w = vmin(v0.w,v1.w)
define <16 x i32> @test_20(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %t0 = icmp slt <16 x i32> %v0, %v1
  %t1 = select <16 x i1> %t0, <16 x i32> %v0, <16 x i32> %v1
  ret <16 x i32> %t1
}

; CHECK: test_21:
; CHECK: v0.w = vmin(v0.w,v1.w)
define <16 x i32> @test_21(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %t0 = icmp sle <16 x i32> %v0, %v1
  %t1 = select <16 x i1> %t0, <16 x i32> %v0, <16 x i32> %v1
  ret <16 x i32> %t1
}

; CHECK: test_22:
; CHECK: v0.w = vmin(v0.w,v1.w)
define <16 x i32> @test_22(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %t0 = icmp sgt <16 x i32> %v0, %v1
  %t1 = select <16 x i1> %t0, <16 x i32> %v1, <16 x i32> %v0
  ret <16 x i32> %t1
}

; CHECK: test_23:
; CHECK: v0.w = vmin(v0.w,v1.w)
define <16 x i32> @test_23(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %t0 = icmp sge <16 x i32> %v0, %v1
  %t1 = select <16 x i1> %t0, <16 x i32> %v1, <16 x i32> %v0
  ret <16 x i32> %t1
}

; maxw

; CHECK: test_24:
; CHECK: v0.w = vmax(v0.w,v1.w)
define <16 x i32> @test_24(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %t0 = icmp slt <16 x i32> %v0, %v1
  %t1 = select <16 x i1> %t0, <16 x i32> %v1, <16 x i32> %v0
  ret <16 x i32> %t1
}

; CHECK: test_25:
; CHECK: v0.w = vmax(v0.w,v1.w)
define <16 x i32> @test_25(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %t0 = icmp sle <16 x i32> %v0, %v1
  %t1 = select <16 x i1> %t0, <16 x i32> %v1, <16 x i32> %v0
  ret <16 x i32> %t1
}

; CHECK: test_26:
; CHECK: v0.w = vmax(v0.w,v1.w)
define <16 x i32> @test_26(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %t0 = icmp sgt <16 x i32> %v0, %v1
  %t1 = select <16 x i1> %t0, <16 x i32> %v0, <16 x i32> %v1
  ret <16 x i32> %t1
}

; CHECK: test_27:
; CHECK: v0.w = vmax(v0.w,v1.w)
define <16 x i32> @test_27(<16 x i32> %v0, <16 x i32> %v1) #0 {
  %t0 = icmp sge <16 x i32> %v0, %v1
  %t1 = select <16 x i1> %t0, <16 x i32> %v0, <16 x i32> %v1
  ret <16 x i32> %t1
}

attributes #0 = { readnone nounwind "target-cpu"="hexagonv62" "target-features"="+hvx,+hvx-length64b" }

