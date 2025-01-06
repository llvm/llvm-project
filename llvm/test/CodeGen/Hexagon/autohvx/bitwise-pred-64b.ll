; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: t00
; CHECK: vand(v{{[0-9:]+}},v{{[0-9:]+}})
define <64 x i8> @t00(<64 x i8> %a0, <64 x i8> %a1) #0 {
  %q0 = trunc <64 x i8> %a0 to <64 x i1>
  %q1 = trunc <64 x i8> %a1 to <64 x i1>
  %q2 = and <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i8>
  ret <64 x i8> %v0
}

; CHECK-LABEL: t01
; CHECK: vor(v{{[0-9:]+}},v{{[0-9:]+}})
define <64 x i8> @t01(<64 x i8> %a0, <64 x i8> %a1) #0 {
  %q0 = trunc <64 x i8> %a0 to <64 x i1>
  %q1 = trunc <64 x i8> %a1 to <64 x i1>
  %q2 = or <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i8>
  ret <64 x i8> %v0
}

; CHECK-LABEL: t02
; CHECK: vxor(v{{[0-9:]+}},v{{[0-9:]+}})
define <64 x i8> @t02(<64 x i8> %a0, <64 x i8> %a1) #0 {
  %q0 = trunc <64 x i8> %a0 to <64 x i1>
  %q1 = trunc <64 x i8> %a1 to <64 x i1>
  %q2 = xor <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i8>
  ret <64 x i8> %v0
}

; CHECK-LABEL: t10
; CHECK: vand(v{{[0-9:]+}},v{{[0-9:]+}})
define <32 x i16> @t10(<32 x i16> %a0, <32 x i16> %a1) #0 {
  %q0 = trunc <32 x i16> %a0 to <32 x i1>
  %q1 = trunc <32 x i16> %a1 to <32 x i1>
  %q2 = and <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i16>
  ret <32 x i16> %v0
}

; CHECK-LABEL: t11
; CHECK: vor(v{{[0-9:]+}},v{{[0-9:]+}})
define <32 x i16> @t11(<32 x i16> %a0, <32 x i16> %a1) #0 {
  %q0 = trunc <32 x i16> %a0 to <32 x i1>
  %q1 = trunc <32 x i16> %a1 to <32 x i1>
  %q2 = or <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i16>
  ret <32 x i16> %v0
}

; CHECK-LABEL: t12
; CHECK: vxor(v{{[0-9:]+}},v{{[0-9:]+}})
define <32 x i16> @t12(<32 x i16> %a0, <32 x i16> %a1) #0 {
  %q0 = trunc <32 x i16> %a0 to <32 x i1>
  %q1 = trunc <32 x i16> %a1 to <32 x i1>
  %q2 = xor <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i16>
  ret <32 x i16> %v0
}

; CHECK-LABEL: t20
; CHECK: vand(v{{[0-9:]+}},v{{[0-9:]+}})
define <16 x i32> @t20(<16 x i32> %a0, <16 x i32> %a1) #0 {
  %q0 = trunc <16 x i32> %a0 to <16 x i1>
  %q1 = trunc <16 x i32> %a1 to <16 x i1>
  %q2 = and <16 x i1> %q0, %q1
  %v0 = zext <16 x i1> %q2 to <16 x i32>
  ret <16 x i32> %v0
}

; CHECK-LABEL: t21
; CHECK: vor(v{{[0-9:]+}},v{{[0-9:]+}})
define <16 x i32> @t21(<16 x i32> %a0, <16 x i32> %a1) #0 {
  %q0 = trunc <16 x i32> %a0 to <16 x i1>
  %q1 = trunc <16 x i32> %a1 to <16 x i1>
  %q2 = or <16 x i1> %q0, %q1
  %v0 = zext <16 x i1> %q2 to <16 x i32>
  ret <16 x i32> %v0
}

; CHECK-LABEL: t22
; CHECK: vxor(v{{[0-9:]+}},v{{[0-9:]+}})
define <16 x i32> @t22(<16 x i32> %a0, <16 x i32> %a1) #0 {
  %q0 = trunc <16 x i32> %a0 to <16 x i1>
  %q1 = trunc <16 x i32> %a1 to <16 x i1>
  %q2 = xor <16 x i1> %q0, %q1
  %v0 = zext <16 x i1> %q2 to <16 x i32>
  ret <16 x i32> %v0
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length64b" }
