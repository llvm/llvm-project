; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: f0:
; CHECK: v1:0.w = vsub(v1:0.w,v1:0.w)
define <128 x i8> @f0() #0 {
  ret <128 x i8> zeroinitializer
}

; CHECK-LABEL: f1:
; CHECK: v1:0.w = vsub(v1:0.w,v1:0.w)
define <64 x i16> @f1() #0 {
  ret <64 x i16> zeroinitializer
}

; CHECK-LABEL: f2:
; CHECK: v1:0.w = vsub(v1:0.w,v1:0.w)
define <32 x i32> @f2() #0 {
  ret <32 x i32> zeroinitializer
}

; CHECK-LABEL: f3:
; CHECK: v1:0.w = vsub(v1:0.w,v1:0.w)
define <256 x i8> @f3() #1 {
  ret <256 x i8> zeroinitializer
}

; CHECK-LABEL: f4:
; CHECK: v1:0.w = vsub(v1:0.w,v1:0.w)
define <128 x i16> @f4() #1 {
  ret <128 x i16> zeroinitializer
}

; CHECK-LABEL: f5:
; CHECK: v1:0.w = vsub(v1:0.w,v1:0.w)
define <64 x i32> @f5() #1 {
  ret <64 x i32> zeroinitializer
}

attributes #0 = { readnone nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
attributes #1 = { readnone nounwind "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }

