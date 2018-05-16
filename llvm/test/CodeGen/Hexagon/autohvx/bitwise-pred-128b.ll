; RUN: llc -march=hexagon < %s | FileCheck %s

; CHECK-LABEL: t00
; CHECK: and(q{{[0-3]}},q{{[0-3]}})
define <128 x i8> @t00(<128 x i8> %a0, <128 x i8> %a1,
                       <128 x i8> %a2, <128 x i8> %a3) #0 {
  %q0 = icmp eq <128 x i8> %a0, %a1
  %q1 = icmp eq <128 x i8> %a2, %a3
  %q2 = and <128 x i1> %q0, %q1
  %v0 = zext <128 x i1> %q2 to <128 x i8>
  ret <128 x i8> %v0
}

; CHECK-LABEL: t01
; CHECK: or(q{{[0-3]}},q{{[0-3]}})
define <128 x i8> @t01(<128 x i8> %a0, <128 x i8> %a1,
                       <128 x i8> %a2, <128 x i8> %a3) #0 {
  %q0 = icmp eq <128 x i8> %a0, %a1
  %q1 = icmp eq <128 x i8> %a2, %a3
  %q2 = or <128 x i1> %q0, %q1
  %v0 = zext <128 x i1> %q2 to <128 x i8>
  ret <128 x i8> %v0
}

; CHECK-LABEL: t02
; CHECK: xor(q{{[0-3]}},q{{[0-3]}})
define <128 x i8> @t02(<128 x i8> %a0, <128 x i8> %a1,
                       <128 x i8> %a2, <128 x i8> %a3) #0 {
  %q0 = icmp eq <128 x i8> %a0, %a1
  %q1 = icmp eq <128 x i8> %a2, %a3
  %q2 = xor <128 x i1> %q0, %q1
  %v0 = zext <128 x i1> %q2 to <128 x i8>
  ret <128 x i8> %v0
}

; CHECK-LABEL: t10
; CHECK: and(q{{[0-3]}},q{{[0-3]}})
define <64 x i16> @t10(<64 x i16> %a0, <64 x i16> %a1,
                       <64 x i16> %a2, <64 x i16> %a3) #0 {
  %q0 = icmp eq <64 x i16> %a0, %a1
  %q1 = icmp eq <64 x i16> %a2, %a3
  %q2 = and <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i16>
  ret <64 x i16> %v0
}

; CHECK-LABEL: t11
; CHECK: or(q{{[0-3]}},q{{[0-3]}})
define <64 x i16> @t11(<64 x i16> %a0, <64 x i16> %a1,
                       <64 x i16> %a2, <64 x i16> %a3) #0 {
  %q0 = icmp eq <64 x i16> %a0, %a1
  %q1 = icmp eq <64 x i16> %a2, %a3
  %q2 = or <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i16>
  ret <64 x i16> %v0
}

; CHECK-LABEL: t12
; CHECK: xor(q{{[0-3]}},q{{[0-3]}})
define <64 x i16> @t12(<64 x i16> %a0, <64 x i16> %a1,
                       <64 x i16> %a2, <64 x i16> %a3) #0 {
  %q0 = icmp eq <64 x i16> %a0, %a1
  %q1 = icmp eq <64 x i16> %a2, %a3
  %q2 = xor <64 x i1> %q0, %q1
  %v0 = zext <64 x i1> %q2 to <64 x i16>
  ret <64 x i16> %v0
}

; CHECK-LABEL: t20
; CHECK: and(q{{[0-3]}},q{{[0-3]}})
define <32 x i32> @t20(<32 x i32> %a0, <32 x i32> %a1,
                       <32 x i32> %a2, <32 x i32> %a3) #0 {
  %q0 = icmp eq <32 x i32> %a0, %a1
  %q1 = icmp eq <32 x i32> %a2, %a3
  %q2 = and <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i32>
  ret <32 x i32> %v0
}

; CHECK-LABEL: t21
; CHECK: or(q{{[0-3]}},q{{[0-3]}})
define <32 x i32> @t21(<32 x i32> %a0, <32 x i32> %a1,
                       <32 x i32> %a2, <32 x i32> %a3) #0 {
  %q0 = icmp eq <32 x i32> %a0, %a1
  %q1 = icmp eq <32 x i32> %a2, %a3
  %q2 = or <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i32>
  ret <32 x i32> %v0
}

; CHECK-LABEL: t22
; CHECK: xor(q{{[0-3]}},q{{[0-3]}})
define <32 x i32> @t22(<32 x i32> %a0, <32 x i32> %a1,
                       <32 x i32> %a2, <32 x i32> %a3) #0 {
  %q0 = icmp eq <32 x i32> %a0, %a1
  %q1 = icmp eq <32 x i32> %a2, %a3
  %q2 = xor <32 x i1> %q0, %q1
  %v0 = zext <32 x i1> %q2 to <32 x i32>
  ret <32 x i32> %v0
}

attributes #0 = { nounwind readnone "target-cpu"="hexagonv60" "target-features"="+hvx,+hvx-length128b" }
