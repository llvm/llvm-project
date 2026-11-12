; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; minb

; CHECK: test_00:
; CHECK: v0.b = vmin(v0.b,v1.b)
define <128 x i8> @test_00(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp slt <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v0, <128 x i8> %v1
  ret <128 x i8> %t1
}

; CHECK: test_01:
; CHECK: v0.b = vmin(v0.b,v1.b)
define <128 x i8> @test_01(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp sle <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v0, <128 x i8> %v1
  ret <128 x i8> %t1
}

; CHECK: test_02:
; CHECK: v0.b = vmin(v0.b,v1.b)
define <128 x i8> @test_02(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp sgt <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v1, <128 x i8> %v0
  ret <128 x i8> %t1
}

; CHECK: test_03:
; CHECK: v0.b = vmin(v0.b,v1.b)
define <128 x i8> @test_03(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp sge <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v1, <128 x i8> %v0
  ret <128 x i8> %t1
}

; maxb

; CHECK: test_04:
; CHECK: v0.b = vmax(v0.b,v1.b)
define <128 x i8> @test_04(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp slt <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v1, <128 x i8> %v0
  ret <128 x i8> %t1
}

; CHECK: test_05:
; CHECK: v0.b = vmax(v0.b,v1.b)
define <128 x i8> @test_05(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp sle <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v1, <128 x i8> %v0
  ret <128 x i8> %t1
}

; CHECK: test_06:
; CHECK: v0.b = vmax(v0.b,v1.b)
define <128 x i8> @test_06(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp sgt <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v0, <128 x i8> %v1
  ret <128 x i8> %t1
}

; CHECK: test_07:
; CHECK: v0.b = vmax(v0.b,v1.b)
define <128 x i8> @test_07(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp sge <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v0, <128 x i8> %v1
  ret <128 x i8> %t1
}

; minub

; CHECK: test_08:
; CHECK: v0.ub = vmin(v0.ub,v1.ub)
define <128 x i8> @test_08(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp ult <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v0, <128 x i8> %v1
  ret <128 x i8> %t1
}

; CHECK: test_09:
; CHECK: v0.ub = vmin(v0.ub,v1.ub)
define <128 x i8> @test_09(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp ule <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v0, <128 x i8> %v1
  ret <128 x i8> %t1
}

; CHECK: test_0a:
; CHECK: v0.ub = vmin(v0.ub,v1.ub)
define <128 x i8> @test_0a(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp ugt <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v1, <128 x i8> %v0
  ret <128 x i8> %t1
}

; CHECK: test_0b:
; CHECK: v0.ub = vmin(v0.ub,v1.ub)
define <128 x i8> @test_0b(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp uge <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v1, <128 x i8> %v0
  ret <128 x i8> %t1
}

; maxub

; CHECK: test_0c:
; CHECK: v0.ub = vmax(v0.ub,v1.ub)
define <128 x i8> @test_0c(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp ult <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v1, <128 x i8> %v0
  ret <128 x i8> %t1
}

; CHECK: test_0d:
; CHECK: v0.ub = vmax(v0.ub,v1.ub)
define <128 x i8> @test_0d(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp ule <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v1, <128 x i8> %v0
  ret <128 x i8> %t1
}

; CHECK: test_0e:
; CHECK: v0.ub = vmax(v0.ub,v1.ub)
define <128 x i8> @test_0e(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp ugt <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v0, <128 x i8> %v1
  ret <128 x i8> %t1
}

; CHECK: test_0f:
; CHECK: v0.ub = vmax(v0.ub,v1.ub)
define <128 x i8> @test_0f(<128 x i8> %v0, <128 x i8> %v1) #0 {
  %t0 = icmp uge <128 x i8> %v0, %v1
  %t1 = select <128 x i1> %t0, <128 x i8> %v0, <128 x i8> %v1
  ret <128 x i8> %t1
}

; minh

; CHECK: test_10:
; CHECK: v0.h = vmin(v0.h,v1.h)
define <64 x i16> @test_10(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp slt <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v0, <64 x i16> %v1
  ret <64 x i16> %t1
}

; CHECK: test_11:
; CHECK: v0.h = vmin(v0.h,v1.h)
define <64 x i16> @test_11(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp sle <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v0, <64 x i16> %v1
  ret <64 x i16> %t1
}

; CHECK: test_12:
; CHECK: v0.h = vmin(v0.h,v1.h)
define <64 x i16> @test_12(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp sgt <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v1, <64 x i16> %v0
  ret <64 x i16> %t1
}

; CHECK: test_13:
; CHECK: v0.h = vmin(v0.h,v1.h)
define <64 x i16> @test_13(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp sge <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v1, <64 x i16> %v0
  ret <64 x i16> %t1
}

; maxh

; CHECK: test_14:
; CHECK: v0.h = vmax(v0.h,v1.h)
define <64 x i16> @test_14(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp slt <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v1, <64 x i16> %v0
  ret <64 x i16> %t1
}

; CHECK: test_15:
; CHECK: v0.h = vmax(v0.h,v1.h)
define <64 x i16> @test_15(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp sle <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v1, <64 x i16> %v0
  ret <64 x i16> %t1
}

; CHECK: test_16:
; CHECK: v0.h = vmax(v0.h,v1.h)
define <64 x i16> @test_16(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp sgt <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v0, <64 x i16> %v1
  ret <64 x i16> %t1
}

; CHECK: test_17:
; CHECK: v0.h = vmax(v0.h,v1.h)
define <64 x i16> @test_17(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp sge <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v0, <64 x i16> %v1
  ret <64 x i16> %t1
}

; minuh

; CHECK: test_18:
; CHECK: v0.uh = vmin(v0.uh,v1.uh)
define <64 x i16> @test_18(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp ult <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v0, <64 x i16> %v1
  ret <64 x i16> %t1
}

; CHECK: test_19:
; CHECK: v0.uh = vmin(v0.uh,v1.uh)
define <64 x i16> @test_19(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp ule <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v0, <64 x i16> %v1
  ret <64 x i16> %t1
}

; CHECK: test_1a:
; CHECK: v0.uh = vmin(v0.uh,v1.uh)
define <64 x i16> @test_1a(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp ugt <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v1, <64 x i16> %v0
  ret <64 x i16> %t1
}

; CHECK: test_1b:
; CHECK: v0.uh = vmin(v0.uh,v1.uh)
define <64 x i16> @test_1b(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp uge <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v1, <64 x i16> %v0
  ret <64 x i16> %t1
}

; maxuh

; CHECK: test_1c:
; CHECK: v0.uh = vmax(v0.uh,v1.uh)
define <64 x i16> @test_1c(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp ult <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v1, <64 x i16> %v0
  ret <64 x i16> %t1
}

; CHECK: test_1d:
; CHECK: v0.uh = vmax(v0.uh,v1.uh)
define <64 x i16> @test_1d(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp ule <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v1, <64 x i16> %v0
  ret <64 x i16> %t1
}

; CHECK: test_1e:
; CHECK: v0.uh = vmax(v0.uh,v1.uh)
define <64 x i16> @test_1e(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp ugt <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v0, <64 x i16> %v1
  ret <64 x i16> %t1
}

; CHECK: test_1f:
; CHECK: v0.uh = vmax(v0.uh,v1.uh)
define <64 x i16> @test_1f(<64 x i16> %v0, <64 x i16> %v1) #0 {
  %t0 = icmp uge <64 x i16> %v0, %v1
  %t1 = select <64 x i1> %t0, <64 x i16> %v0, <64 x i16> %v1
  ret <64 x i16> %t1
}

; minw

; CHECK: test_20:
; CHECK: v0.w = vmin(v0.w,v1.w)
define <32 x i32> @test_20(<32 x i32> %v0, <32 x i32> %v1) #0 {
  %t0 = icmp slt <32 x i32> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i32> %v0, <32 x i32> %v1
  ret <32 x i32> %t1
}

; CHECK: test_21:
; CHECK: v0.w = vmin(v0.w,v1.w)
define <32 x i32> @test_21(<32 x i32> %v0, <32 x i32> %v1) #0 {
  %t0 = icmp sle <32 x i32> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i32> %v0, <32 x i32> %v1
  ret <32 x i32> %t1
}

; CHECK: test_22:
; CHECK: v0.w = vmin(v0.w,v1.w)
define <32 x i32> @test_22(<32 x i32> %v0, <32 x i32> %v1) #0 {
  %t0 = icmp sgt <32 x i32> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i32> %v1, <32 x i32> %v0
  ret <32 x i32> %t1
}

; CHECK: test_23:
; CHECK: v0.w = vmin(v0.w,v1.w)
define <32 x i32> @test_23(<32 x i32> %v0, <32 x i32> %v1) #0 {
  %t0 = icmp sge <32 x i32> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i32> %v1, <32 x i32> %v0
  ret <32 x i32> %t1
}

; maxw

; CHECK: test_24:
; CHECK: v0.w = vmax(v0.w,v1.w)
define <32 x i32> @test_24(<32 x i32> %v0, <32 x i32> %v1) #0 {
  %t0 = icmp slt <32 x i32> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i32> %v1, <32 x i32> %v0
  ret <32 x i32> %t1
}

; CHECK: test_25:
; CHECK: v0.w = vmax(v0.w,v1.w)
define <32 x i32> @test_25(<32 x i32> %v0, <32 x i32> %v1) #0 {
  %t0 = icmp sle <32 x i32> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i32> %v1, <32 x i32> %v0
  ret <32 x i32> %t1
}

; CHECK: test_26:
; CHECK: v0.w = vmax(v0.w,v1.w)
define <32 x i32> @test_26(<32 x i32> %v0, <32 x i32> %v1) #0 {
  %t0 = icmp sgt <32 x i32> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i32> %v0, <32 x i32> %v1
  ret <32 x i32> %t1
}

; CHECK: test_27:
; CHECK: v0.w = vmax(v0.w,v1.w)
define <32 x i32> @test_27(<32 x i32> %v0, <32 x i32> %v1) #0 {
  %t0 = icmp sge <32 x i32> %v0, %v1
  %t1 = select <32 x i1> %t0, <32 x i32> %v0, <32 x i32> %v1
  ret <32 x i32> %t1
}

attributes #0 = { readnone nounwind "target-cpu"="hexagonv62" "target-features"="+hvx,+hvx-length128b" }

