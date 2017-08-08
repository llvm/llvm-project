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
