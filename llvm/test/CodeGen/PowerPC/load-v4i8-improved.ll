; RUN: llc -mcpu=pwr8 -mtriple=powerpc64le-unknown-linux-gnu < %s | FileCheck \
; RUN:   -implicit-check-not vmrg -implicit-check-not=vperm %s
; RUN: llc -mcpu=pwr8 -mtriple=powerpc64-unknown-linux-gnu < %s | FileCheck \
; RUN:   -implicit-check-not vmrg -implicit-check-not=vperm %s \
; RUN:   --check-prefix=CHECK-BE

define <16 x i8> @test(i32* %s, i32* %t) {
entry:
  %0 = bitcast i32* %s to <4 x i8>*
  %1 = load <4 x i8>, <4 x i8>* %0, align 4
  %2 = shufflevector <4 x i8> %1, <4 x i8> undef, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3, i32 0, i32 1, i32 2, i32 3>
  ret <16 x i8> %2
; CHECK-LABEL: test
; CHECK: lwz [[GPR:[0-9]+]], 0(3)
; CHECK: mtvsrd [[VSR:[0-9]+]], [[GPR]]
; CHECK: xxswapd  [[SWP:[0-9]+]], [[VSR]]
; CHECK: xxspltw 34, [[SWP]], 3
; CHECK-BE-LABEL: test
; CHECK-BE: lwz [[GPR:[0-9]+]], 0(3)
; CHECK-BE: sldi [[SHL:[0-9]+]], [[GPR]], 32
; CHECK-BE: mtvsrd [[VSR:[0-9]+]], [[SHL]]
; CHECK-BE: xxspltw 34, [[VSR]], 0
}
