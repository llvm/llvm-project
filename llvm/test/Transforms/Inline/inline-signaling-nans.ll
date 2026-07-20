; RUN: opt < %s -passes=inline -S | FileCheck %s

; Check the inliner combines signaling_nans attribute so that the calling
; function has this attribute if it or the called functions have it.
 
define internal float @not_signaling(float %x, float %y) #0 {
  %sum = fadd float %x, %y
  ret float %sum
}

define internal float @signaling(float %x, float %y) #1 {
  %mul = fmul float %x, %y
  ret float %mul
}

define float @all_not_signaling(float %x, float %y, float %z) #0 {
  %v = call float @not_signaling(float %x, float %y)
  %res = fmul float %v, %z
  ret float %res
}
; CHECK: define float @all_not_signaling({{.*}}) [[NOT_SIGNALING:#[0-9]+]] {

define float @all_signaling(float %x, float %y, float %z) #1 {
  %v = call float @signaling(float %x, float %y)
  %res = fmul float %v, %z
  ret float %res
}
; CHECK: define float @all_signaling({{.*}}) [[SIGNALING:#[0-9]+]] {

define float @to_not_signaling(float %x, float %y, float %z) #0 {
  %v = call float @signaling(float %x, float %y)
  %res = fmul float %v, %z
  ret float %res
}
; CHECK: define float @to_not_signaling({{.*}}) [[SIGNALING:#[0-9]+]] {

define float @to_signaling(float %x, float %y, float %z) #1 {
  %v = call float @not_signaling(float %x, float %y)
  %res = fmul float %v, %z
  ret float %res
}
; CHECK: define float @to_signaling({{.*}}) [[SIGNALING:#[0-9]+]] {

attributes #0 = { nounwind }
; CHECK: attributes [[NOT_SIGNALING]] = { nounwind }
attributes #1 = { nounwind signaling_nans }
; CHECK: attributes [[SIGNALING]] = { nounwind signaling_nans }
