; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: test0000:
; CHECK: v0.h = vasl(v0.h,r0)
define <32 x i16> @test0000(<32 x i16> %a0, i16 %a1) #0 {
  %b0 = insertelement <32 x i16> zeroinitializer, i16 %a1, i32 0
  %b1 = shufflevector <32 x i16> %b0, <32 x i16> undef, <32 x i32> zeroinitializer

  %v0 = shl <32 x i16> %a0, %b1
  ret <32 x i16> %v0
}

; CHECK-LABEL: test0001:
; CHECK: v0.h = vasr(v0.h,r0)
define <32 x i16> @test0001(<32 x i16> %a0, i16 %a1) #0 {
  %b0 = insertelement <32 x i16> zeroinitializer, i16 %a1, i32 0
  %b1 = shufflevector <32 x i16> %b0, <32 x i16> undef, <32 x i32> zeroinitializer
  %v0 = ashr <32 x i16> %a0, %b1
  ret <32 x i16> %v0
}

; CHECK-LABEL: test0002:
; CHECK: v0.uh = vlsr(v0.uh,r0)
define <32 x i16> @test0002(<32 x i16> %a0, i16 %a1) #0 {
  %b0 = insertelement <32 x i16> zeroinitializer, i16 %a1, i32 0
  %b1 = shufflevector <32 x i16> %b0, <32 x i16> undef, <32 x i32> zeroinitializer
  %v0 = lshr <32 x i16> %a0, %b1
  ret <32 x i16> %v0
}

; CHECK-LABEL: test0010:
; CHECK: v0.w = vasl(v0.w,r0)
define <16 x i32> @test0010(<16 x i32> %a0, i32 %a1) #0 {
  %b0 = insertelement <16 x i32> zeroinitializer, i32 %a1, i32 0
  %b1 = shufflevector <16 x i32> %b0, <16 x i32> undef, <16 x i32> zeroinitializer
  %v0 = shl <16 x i32> %a0, %b1
  ret <16 x i32> %v0
}

; CHECK-LABEL: test0011:
; CHECK: v0.w = vasr(v0.w,r0)
define <16 x i32> @test0011(<16 x i32> %a0, i32 %a1) #0 {
  %b0 = insertelement <16 x i32> zeroinitializer, i32 %a1, i32 0
  %b1 = shufflevector <16 x i32> %b0, <16 x i32> undef, <16 x i32> zeroinitializer
  %v0 = ashr <16 x i32> %a0, %b1
  ret <16 x i32> %v0
}

; CHECK-LABEL: test0012:
; CHECK: v0.uw = vlsr(v0.uw,r0)
define <16 x i32> @test0012(<16 x i32> %a0, i32 %a1) #0 {
  %b0 = insertelement <16 x i32> zeroinitializer, i32 %a1, i32 0
  %b1 = shufflevector <16 x i32> %b0, <16 x i32> undef, <16 x i32> zeroinitializer
  %v0 = lshr <16 x i32> %a0, %b1
  ret <16 x i32> %v0
}

; CHECK-LABEL: test0013:
; CHECK: v0.w += vasl(v1.w,r0)
define <16 x i32> @test0013(<16 x i32> %a0, <16 x i32> %a1, i32 %a2) #0 {
  %b0 = insertelement <16 x i32> zeroinitializer, i32 %a2, i32 0
  %b1 = shufflevector <16 x i32> %b0, <16 x i32> undef, <16 x i32> zeroinitializer
  %v0 = shl <16 x i32> %a1, %b1
  %v1 = add <16 x i32> %a0, %v0
  ret <16 x i32> %v1
}

; CHECK-LABEL: test0014:
; CHECK: v0.w += vasr(v1.w,r0)
define <16 x i32> @test0014(<16 x i32> %a0, <16 x i32> %a1, i32 %a2) #0 {
  %b0 = insertelement <16 x i32> zeroinitializer, i32 %a2, i32 0
  %b1 = shufflevector <16 x i32> %b0, <16 x i32> undef, <16 x i32> zeroinitializer
  %v0 = ashr <16 x i32> %a1, %b1
  %v1 = add <16 x i32> %a0, %v0
  ret <16 x i32> %v1
}

; CHECK-LABEL: test0020:
; CHECK: v0.h = vasl(v0.h,v1.h)
define <32 x i16> @test0020(<32 x i16> %a0, <32 x i16> %a1) #0 {
  %v0 = shl <32 x i16> %a0, %a1
  ret <32 x i16> %v0
}

; CHECK-LABEL: test0021:
; CHECK: v0.h = vasr(v0.h,v1.h)
define <32 x i16> @test0021(<32 x i16> %a0, <32 x i16> %a1) #0 {
  %v0 = ashr <32 x i16> %a0, %a1
  ret <32 x i16> %v0
}

; CHECK-LABEL: test0022:
; CHECK: v0.h = vlsr(v0.h,v1.h)
define <32 x i16> @test0022(<32 x i16> %a0, <32 x i16> %a1) #0 {
  %v0 = lshr <32 x i16> %a0, %a1
  ret <32 x i16> %v0
}

; CHECK-LABEL: test0030:
; CHECK: v0.w = vasl(v0.w,v1.w)
define <16 x i32> @test0030(<16 x i32> %a0, <16 x i32> %a1) #0 {
  %v0 = shl <16 x i32> %a0, %a1
  ret <16 x i32> %v0
}

; CHECK-LABEL: test0031:
; CHECK: v0.w = vasr(v0.w,v1.w)
define <16 x i32> @test0031(<16 x i32> %a0, <16 x i32> %a1) #0 {
  %v0 = ashr <16 x i32> %a0, %a1
  ret <16 x i32> %v0
}

; CHECK-LABEL: test0032:
; CHECK: v0.w = vlsr(v0.w,v1.w)
define <16 x i32> @test0032(<16 x i32> %a0, <16 x i32> %a1) #0 {
  %v0 = lshr <16 x i32> %a0, %a1
  ret <16 x i32> %v0
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }

