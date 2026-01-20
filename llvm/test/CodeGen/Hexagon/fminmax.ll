; RUN: llc -mtriple=hexagon < %s | FileCheck %s

target datalayout = "e-m:e-p:32:32:32-a:0-n16:32-i64:64:64-i32:32:32-i16:16:16-i1:8:8-f32:32:32-f64:64:64-v32:32:32-v64:64:64-v512:512:512-v1024:1024:1024-v2048:2048:2048"
target triple = "hexagon"

; CHECK-LABEL: cfminf
; CHECK: jump fminf
define float @cfminf(float %x, float %y) #0 {
entry:
  %call = tail call float @fminf(float %x, float %y) #1
  ret float %call
}

; CHECK-LABEL: cfmaxf
; CHECK: jump fmaxf
define float @cfmaxf(float %x, float %y) #0 {
entry:
  %call = tail call float @fmaxf(float %x, float %y) #1
  ret float %call
}

; CHECK-LABEL: minnum
; CHECK: jump fminf
define float @minnum(float %x, float %y) #0 {
entry:
  %call = tail call float @llvm.minnum.f32(float %x, float %y) #1
  ret float %call
}

; CHECK-LABEL: maxnum
; CHECK: jump fmaxf
define float @maxnum(float %x, float %y) #0 {
entry:
  %call = tail call float @llvm.maxnum.f32(float %x, float %y) #1
  ret float %call
}

; CHECK-LABEL: minimumnum
; CHECK: sfmin
define float @minimumnum(float %x, float %y) #0 {
entry:
  %call = tail call float @llvm.minimumnum.f32(float %x, float %y) #1
  ret float %call
}

; CHECK-LABEL: maximumnum
; CHECK: sfmax
define float @maximumnum(float %x, float %y) #0 {
entry:
  %call = tail call float @llvm.maximumnum.f32(float %x, float %y) #1
  ret float %call
}


declare float @fminf(float, float) #0
declare float @fmaxf(float, float) #0

attributes #0 = { nounwind readnone "disable-tail-calls"="false" "less-precise-fpmad"="false" "frame-pointer"="all" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone }

