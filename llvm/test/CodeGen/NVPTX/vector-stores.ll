; RUN: llc < %s -march=nvptx64 -mcpu=sm_20 | FileCheck %s
; RUN: %if ptxas %{ llc < %s -march=nvptx64 -mcpu=sm_20 | %ptxas-verify %}

; CHECK-LABEL: .visible .func foo1
; CHECK: st.v2.f32
define void @foo1(<2 x float> %val, ptr %ptr) {
  store <2 x float> %val, ptr %ptr
  ret void
}

; CHECK-LABEL: .visible .func foo2
; CHECK: st.v4.f32
define void @foo2(<4 x float> %val, ptr %ptr) {
  store <4 x float> %val, ptr %ptr
  ret void
}

; CHECK-LABEL: .visible .func foo3
; CHECK: st.v2.u32
define void @foo3(<2 x i32> %val, ptr %ptr) {
  store <2 x i32> %val, ptr %ptr
  ret void
}

; CHECK-LABEL: .visible .func foo4
; CHECK: st.v4.u32
define void @foo4(<4 x i32> %val, ptr %ptr) {
  store <4 x i32> %val, ptr %ptr
  ret void
}

; CHECK-LABEL: .visible .func v16i8
define void @v16i8(ptr %a, ptr %b) {
; CHECK: ld.v4.u32
; CHECK: st.v4.u32
  %v = load <16 x i8>, ptr %a
  store <16 x i8> %v, ptr %b
  ret void
}
