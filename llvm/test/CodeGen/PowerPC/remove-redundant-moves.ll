; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-BE
define double @test1(<2 x i64> %a) {
entry:
; CHECK-LABEL: test1
; CHECK: xxswapd [[SW:[0-9]+]], 34
; CHECK: xscvsxddp 1, [[SW]]
; CHECK-BE-LABEL: test1
; CHECK-BE: xxlor [[CP:[0-9]+]], 34, 34
; CHECK-BE: xscvsxddp 1, [[CP]]
  %0 = extractelement <2 x i64> %a, i32 0
  %1 = sitofp i64 %0 to double
  ret double %1
}

define double @test2(<2 x i64> %a) {
entry:
; CHECK-LABEL: test2
; CHECK: xxlor [[CP:[0-9]+]], 34, 34
; CHECK: xscvsxddp 1, [[CP]]
; CHECK-BE-LABEL: test2
; CHECK-BE: xxswapd [[SW:[0-9]+]], 34
; CHECK-BE: xscvsxddp 1, [[SW]]
  %0 = extractelement <2 x i64> %a, i32 1
  %1 = sitofp i64 %0 to double
  ret double %1
}

define float @test1f(<2 x i64> %a) {
entry:
; CHECK-LABEL: test1f
; CHECK: xxswapd [[SW:[0-9]+]], 34
; CHECK: xscvsxdsp 1, [[SW]]
; CHECK-BE-LABEL: test1f
; CHECK-BE: xxlor [[CP:[0-9]+]], 34, 34
; CHECK-BE: xscvsxdsp 1, [[CP]]
  %0 = extractelement <2 x i64> %a, i32 0
  %1 = sitofp i64 %0 to float
  ret float %1
}

define float @test2f(<2 x i64> %a) {
entry:
; CHECK-LABEL: test2f
; CHECK: xxlor [[CP:[0-9]+]], 34, 34
; CHECK: xscvsxdsp 1, [[CP]]
; CHECK-BE-LABEL: test2f
; CHECK-BE: xxswapd [[SW:[0-9]+]], 34
; CHECK-BE: xscvsxdsp 1, [[SW]]
  %0 = extractelement <2 x i64> %a, i32 1
  %1 = sitofp i64 %0 to float
  ret float %1
}

define double @test1u(<2 x i64> %a) {
entry:
; CHECK-LABEL: test1u
; CHECK: xxswapd [[SW:[0-9]+]], 34
; CHECK: xscvuxddp 1, [[SW]]
; CHECK-BE-LABEL: test1u
; CHECK-BE: xxlor [[CP:[0-9]+]], 34, 34
; CHECK-BE: xscvuxddp 1, [[CP]]
  %0 = extractelement <2 x i64> %a, i32 0
  %1 = uitofp i64 %0 to double
  ret double %1
}

define double @test2u(<2 x i64> %a) {
entry:
; CHECK-LABEL: test2u
; CHECK: xxlor [[CP:[0-9]+]], 34, 34
; CHECK: xscvuxddp 1, [[CP]]
; CHECK-BE-LABEL: test2u
; CHECK-BE: xxswapd [[SW:[0-9]+]], 34
; CHECK-BE: xscvuxddp 1, [[SW]]
  %0 = extractelement <2 x i64> %a, i32 1
  %1 = uitofp i64 %0 to double
  ret double %1
}

define float @test1fu(<2 x i64> %a) {
entry:
; CHECK-LABEL: test1fu
; CHECK: xxswapd [[SW:[0-9]+]], 34
; CHECK: xscvuxdsp 1, [[SW]]
; CHECK-BE-LABEL: test1fu
; CHECK-BE: xxlor [[CP:[0-9]+]], 34, 34
; CHECK-BE: xscvuxdsp 1, [[CP]]
  %0 = extractelement <2 x i64> %a, i32 0
  %1 = uitofp i64 %0 to float
  ret float %1
}

define float @test2fu(<2 x i64> %a) {
entry:
; CHECK-LABEL: test2fu
; CHECK: xxlor [[CP:[0-9]+]], 34, 34
; CHECK: xscvuxdsp 1, [[CP]]
; CHECK-BE-LABEL: test2fu
; CHECK-BE: xxswapd [[SW:[0-9]+]], 34
; CHECK-BE: xscvuxdsp 1, [[SW]]
  %0 = extractelement <2 x i64> %a, i32 1
  %1 = uitofp i64 %0 to float
  ret float %1
}

define float @conv2fltTesti0(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2fltTesti0
; CHECK: xxspltw [[SW:[0-9]+]], 34, 3
; CHECK: xvcvsxwsp [[SW]], [[SW]]
; CHECK: xscvspdpn 1, [[SW]]
; CHECK-BE-LABEL: conv2fltTesti0
; CHECK-BE: xxspltw [[CP:[0-9]+]], 34, 0
; CHECK-BE: xvcvsxwsp [[CP]], [[CP]]
; CHECK-BE: xscvspdpn 1, [[CP]]
  %vecext = extractelement <4 x i32> %a, i32 0
  %conv = sitofp i32 %vecext to float
  ret float %conv
}

define float @conv2fltTesti1(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2fltTesti1
; CHECK: xxspltw [[SW:[0-9]+]], 34, 2
; CHECK: xvcvsxwsp [[SW]], [[SW]]
; CHECK: xscvspdpn 1, [[SW]]
; CHECK-BE-LABEL: conv2fltTesti1
; CHECK-BE: xxspltw [[CP:[0-9]+]], 34, 1
; CHECK-BE: xvcvsxwsp [[CP]], [[CP]]
; CHECK-BE: xscvspdpn 1, [[CP]]
  %vecext = extractelement <4 x i32> %a, i32 1
  %conv = sitofp i32 %vecext to float
  ret float %conv
}

define float @conv2fltTesti2(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2fltTesti2
; CHECK: xxspltw [[SW:[0-9]+]], 34, 1
; CHECK: xvcvsxwsp [[SW]], [[SW]]
; CHECK: xscvspdpn 1, [[SW]]
; CHECK-BE-LABEL: conv2fltTesti2
; CHECK-BE: xxspltw [[CP:[0-9]+]], 34, 2
; CHECK-BE: xvcvsxwsp [[CP]], [[CP]]
; CHECK-BE: xscvspdpn 1, [[CP]]
  %vecext = extractelement <4 x i32> %a, i32 2
  %conv = sitofp i32 %vecext to float
  ret float %conv
}

define float @conv2fltTesti3(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2fltTesti3
; CHECK: xxspltw [[SW:[0-9]+]], 34, 0
; CHECK: xvcvsxwsp [[SW]], [[SW]]
; CHECK: xscvspdpn 1, [[SW]]
; CHECK-BE-LABEL: conv2fltTesti3
; CHECK-BE: xxspltw [[CP:[0-9]+]], 34, 3
; CHECK-BE: xvcvsxwsp [[CP]], [[CP]]
; CHECK-BE: xscvspdpn 1, [[CP]]
  %vecext = extractelement <4 x i32> %a, i32 3
  %conv = sitofp i32 %vecext to float
  ret float %conv
}

; verify we don't crash for variable elem extract
define float @conv2fltTestiVar(<4 x i32> %a, i32 zeroext %elem) {
entry:
  %vecext = extractelement <4 x i32> %a, i32 %elem
  %conv = sitofp i32 %vecext to float
  ret float %conv
}

define double @conv2dblTesti0(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2dblTesti0
; CHECK: xxspltw [[SW:[0-9]+]], 34, 3
; CHECK: xvcvsxwdp 1, [[SW]]
; CHECK-BE-LABEL: conv2dblTesti0
; CHECK-BE: xxspltw [[CP:[0-9]+]], 34, 0
; CHECK-BE: xvcvsxwdp 1, [[CP]]
  %vecext = extractelement <4 x i32> %a, i32 0
  %conv = sitofp i32 %vecext to double
  ret double %conv
}

define double @conv2dblTesti1(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2dblTesti1
; CHECK: xxspltw [[SW:[0-9]+]], 34, 2
; CHECK: xvcvsxwdp 1, [[SW]]
; CHECK-BE-LABEL: conv2dblTesti1
; CHECK-BE: xxspltw [[CP:[0-9]+]], 34, 1
; CHECK-BE: xvcvsxwdp 1, [[CP]]
  %vecext = extractelement <4 x i32> %a, i32 1
  %conv = sitofp i32 %vecext to double
  ret double %conv
}

define double @conv2dblTesti2(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2dblTesti2
; CHECK: xxspltw [[SW:[0-9]+]], 34, 1
; CHECK: xvcvsxwdp 1, [[SW]]
; CHECK-BE-LABEL: conv2dblTesti2
; CHECK-BE: xxspltw [[CP:[0-9]+]], 34, 2
; CHECK-BE: xvcvsxwdp 1, [[CP]]
  %vecext = extractelement <4 x i32> %a, i32 2
  %conv = sitofp i32 %vecext to double
  ret double %conv
}

define double @conv2dblTesti3(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2dblTesti3
; CHECK: xxspltw [[SW:[0-9]+]], 34, 0
; CHECK: xvcvsxwdp 1, [[SW]]
; CHECK-BE-LABEL: conv2dblTesti3
; CHECK-BE: xxspltw [[CP:[0-9]+]], 34, 3
; CHECK-BE: xvcvsxwdp 1, [[CP]]
  %vecext = extractelement <4 x i32> %a, i32 3
  %conv = sitofp i32 %vecext to double
  ret double %conv
}

; verify we don't crash for variable elem extract
define double @conv2dblTestiVar(<4 x i32> %a, i32 zeroext %elem) {
entry:
  %vecext = extractelement <4 x i32> %a, i32 %elem
  %conv = sitofp i32 %vecext to double
  ret double %conv
}
