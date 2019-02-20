; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: t00
; CHECK: vand(v{{[0-9:]+}},v{{[0-9:]+}})
define <128 x i8> @t00(<128 x i8> %a0, <128 x i8> %a1) #0 {
  %q0 = trunc <128 x i8> %a0 to <128 x i1>
  %q1 = trunc <128 x i8> %a1 to <128 x i1>
  %q2 = and <128 x i1> %q0, %q1
  %v0 = zext <128 x i1> %q2 to <128 x i8>
  ret <128 x i8> %v0
}

declare <1024 x i1> @llvm.hexagon.vandvrt.128B(<128 x i8>, i32)

; CHECK-LABEL: t01
; CHECK: vor(v{{[0-9:]+}},v{{[0-9:]+}})
define <128 x i8> @t01(<128 x i8> %a0, <128 x i8> %a1) #0 {
  %q0 = trunc <128 x i8> %a0 to <128 x i1>
  %q1 = trunc <128 x i8> %a1 to <128 x i1>
  %q2 = or <128 x i1> %q0, %q1
  %v0 = zext <128 x i1> %q2 to <128 x i8>
  ret <128 x i8> %v0
}

; CHECK-LABEL: t02
; CHECK: vxor(v{{[0-9:]+}},v{{[0-9:]+}})
define <128 x i8> @t02(<128 x i8> %a0, <128 x i8> %a1) #0 {
  %q0 = trunc <128 x i8> %a0 to <128 x i1>
  %q1 = trunc <128 x i8> %a1 to <128 x i1>
  %q2 = xor <128 x i1> %q0, %q1
  %v0 = zext <128 x i1> %q2 to <128 x i8>
  ret <128 x i8> %v0
}

; CHECK-LABEL: t10
; CHECK: vand(v{{[0-9:]+}},v{{[0-9:]+}})
define <64 x i16> @t10(<64 x i16> %a0, <64 x i16> %a1) #0 {
  %q0 = trunc <64 x i16> %a0 to <64 x i1>
  %q1 = trunc <64 x i16> %a1 to <64 x i1>
  %q2 = and <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i16>
  ret <64 x i16> %v0
}

; CHECK-LABEL: t11
; CHECK: vor(v{{[0-9:]+}},v{{[0-9:]+}})
define <64 x i16> @t11(<64 x i16> %a0, <64 x i16> %a1) #0 {
  %q0 = trunc <64 x i16> %a0 to <64 x i1>
  %q1 = trunc <64 x i16> %a1 to <64 x i1>
  %q2 = or <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i16>
  ret <64 x i16> %v0
}

; CHECK-LABEL: t12
; CHECK: vxor(v{{[0-9:]+}},v{{[0-9:]+}})
define <64 x i16> @t12(<64 x i16> %a0, <64 x i16> %a1) #0 {
  %q0 = trunc <64 x i16> %a0 to <64 x i1>
  %q1 = trunc <64 x i16> %a1 to <64 x i1>
  %q2 = xor <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i16>
  ret <64 x i16> %v0
}

; CHECK-LABEL: t20
; CHECK: vand(v{{[0-9:]+}},v{{[0-9:]+}})
define <32 x i32> @t20(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %q0 = trunc <32 x i32> %a0 to <32 x i1>
  %q1 = trunc <32 x i32> %a1 to <32 x i1>
  %q2 = and <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i32>
  ret <32 x i32> %v0
}

; CHECK-LABEL: t21
; CHECK: vor(v{{[0-9:]+}},v{{[0-9:]+}})
define <32 x i32> @t21(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %q0 = trunc <32 x i32> %a0 to <32 x i1>
  %q1 = trunc <32 x i32> %a1 to <32 x i1>
  %q2 = or <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i32>
  ret <32 x i32> %v0
}

; CHECK-LABEL: t22
; CHECK: vxor(v{{[0-9:]+}},v{{[0-9:]+}})
define <32 x i32> @t22(<32 x i32> %a0, <32 x i32> %a1) #0 {
  %q0 = trunc <32 x i32> %a0 to <32 x i1>
  %q1 = trunc <32 x i32> %a1 to <32 x i1>
  %q2 = xor <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i32>
  ret <32 x i32> %v0
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
