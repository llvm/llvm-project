; RUN: llc -mcpu=pwr9 -mtriple=powerpc64-ibm-aix-xcoff -vec-extabi \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-64
; RUN: llc -mcpu=pwr9 -mtriple=powerpc-ibm-aix-xcoff -vec-extabi \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-32

define <4 x float> @_Z7testInsILj0ELj0EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj0EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj1EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj1EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 5, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj2EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj2EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 6, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj3EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj3EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 7, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj0EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj0EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj1EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj1EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 4
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj2EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj2EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj3EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj3EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 7, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj0EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj0EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj1EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj1EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj2EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj2EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj3EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj3EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 7, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj0EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj0EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj1EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj1EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj2EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj2EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj3EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj3EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x float> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj0EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj0EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj1EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj1EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 0
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 5, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj2EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj2EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 6, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj3EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj3EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 7, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj0EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj0EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj1EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj1EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 4
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj2EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj2EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj3EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj3EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 7, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj0EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj0EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj1EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj1EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 8
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj2EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj2EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj3EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj3EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 7, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj0EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj0EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj1EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj1EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 12
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj2EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj2EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj3EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj3EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x i32> %vecins
}

define float @_Z13testUiToFpExtILj0EEfDv4_j(<4 x i32> %a) {
entry:
; CHECK-64-LABEL: _Z13testUiToFpExtILj0EEfDv4_j
; CHECK-64: xxextractuw 0, 34, 0
; CHECK-64: xscvuxdsp 1, 0
; CHECK-32-LABEL: _Z13testUiToFpExtILj0EEfDv4_j
; CHECK-32: lfiwzx 0, 0, 3
; CHECK-32: xscvuxdsp 1, 0
  %vecext = extractelement <4 x i32> %a, i32 0
  %conv = uitofp i32 %vecext to float
  ret float %conv
}

define float @_Z13testUiToFpExtILj1EEfDv4_j(<4 x i32> %a) {
entry:
; CHECK-64-LABEL: _Z13testUiToFpExtILj1EEfDv4_j
; CHECK-64: xxextractuw 0, 34, 4
; CHECK-64: xscvuxdsp 1, 0
; CHECK-32-LABEL: _Z13testUiToFpExtILj1EEfDv4_j
; CHECK-32: lfiwzx 0, 0, 3
; CHECK-32: xscvuxdsp 1, 0
  %vecext = extractelement <4 x i32> %a, i32 1
  %conv = uitofp i32 %vecext to float
  ret float %conv
}

define float @_Z13testUiToFpExtILj2EEfDv4_j(<4 x i32> %a) {
entry:
; CHECK-64-LABEL: _Z13testUiToFpExtILj2EEfDv4_j
; CHECK-64: xxextractuw 0, 34, 8
; CHECK-64: xscvuxdsp 1, 0
; CHECK-32-LABEL: _Z13testUiToFpExtILj2EEfDv4_j
; CHECK-32: lfiwzx 0, 0, 3
; CHECK-32: xscvuxdsp 1, 0
  %vecext = extractelement <4 x i32> %a, i32 2
  %conv = uitofp i32 %vecext to float
  ret float %conv
}

define float @_Z13testUiToFpExtILj3EEfDv4_j(<4 x i32> %a) {
entry:
; CHECK-64-LABEL: _Z13testUiToFpExtILj3EEfDv4_j
; CHECK-64: xxextractuw 0, 34, 12
; CHECK-64: xscvuxdsp 1, 0
; CHECK-32-LABEL: _Z13testUiToFpExtILj3EEfDv4_j
; CHECK-32: lfiwzx 0, 0, 3
; CHECK-32: xscvuxdsp 1, 0
  %vecext = extractelement <4 x i32> %a, i32 3
  %conv = uitofp i32 %vecext to float
  ret float %conv
}

; Verify we generate optimal code for unsigned vector int elem extract followed
; by conversion to double

define double @conv2dlbTestui0(<4 x i32> %a) {
entry:
; CHECK-64-LABEL: conv2dlbTestui0
; CHECK-64: xxextractuw [[CP64:[0-9]+]], 34, 0
; CHECK-64: xscvuxddp 1, [[CP64]]
; CHECK-32-LABEL: conv2dlbTestui0
; CHECK-32: lfiwzx [[CP32:[0-9]+]], 0, 3
; CHECK-32: xscvuxddp 1, [[CP32]]
  %0 = extractelement <4 x i32> %a, i32 0
  %1 = uitofp i32 %0 to double
  ret double %1
}

define double @conv2dlbTestui1(<4 x i32> %a) {
entry:
; CHECK-64-LABEL: conv2dlbTestui1
; CHECK-64: xxextractuw [[CP64:[0-9]+]], 34, 4
; CHECK-64: xscvuxddp 1, [[CP64]]
; CHECK-32-LABEL: conv2dlbTestui1
; CHECK-32: lfiwzx [[CP32:[0-9]+]], 0, 3
; CHECK-32: xscvuxddp 1, [[CP32]]
  %0 = extractelement <4 x i32> %a, i32 1
  %1 = uitofp i32 %0 to double
  ret double %1
}

define double @conv2dlbTestui2(<4 x i32> %a) {
entry:
; CHECK-64-LABEL: conv2dlbTestui2
; CHECK-64: xxextractuw [[CP64:[0-9]+]], 34, 8
; CHECK-64: xscvuxddp 1, [[CP64]]
; CHECK-32-LABEL: conv2dlbTestui2
; CHECK-32: lfiwzx [[CP32:[0-9]+]], 0, 3
; CHECK-32: xscvuxddp 1, [[CP32]]
  %0 = extractelement <4 x i32> %a, i32 2
  %1 = uitofp i32 %0 to double
  ret double %1
}

define double @conv2dlbTestui3(<4 x i32> %a) {
entry:
; CHECK-64-LABEL: conv2dlbTestui3
; CHECK-64: xxextractuw [[CP64:[0-9]+]], 34, 12
; CHECK-64: xscvuxddp 1, [[CP64]]
; CHECK-32-LABEL: conv2dlbTestui3
; CHECK-32: lfiwzx [[CP32:[0-9]+]], 0, 3
; CHECK-32: xscvuxddp 1, [[CP32]]
  %0 = extractelement <4 x i32> %a, i32 3
  %1 = uitofp i32 %0 to double
  ret double %1
}

; verify we don't crash for variable elem extract
define double @conv2dlbTestuiVar(<4 x i32> %a, i32 zeroext %elem) {
entry:
  %vecext = extractelement <4 x i32> %a, i32 %elem
  %conv = uitofp i32 %vecext to double
  ret double %conv
}

define <4 x float> @_Z10testInsEltILj0EDv4_ffET0_S1_T1_(<4 x float> %a, float %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj0EDv4_ffET0_S1_T1_
; CHECK: xscvdpspn 0, 1
; CHECK: xxsldwi 0, 0, 0, 3
; CHECK: xxinsertw 34, 0, 0
  %vecins = insertelement <4 x float> %a, float %b, i32 0
  ret <4 x float> %vecins
}

define <4 x float> @_Z10testInsEltILj1EDv4_ffET0_S1_T1_(<4 x float> %a, float %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj1EDv4_ffET0_S1_T1_
; CHECK: xscvdpspn 0, 1
; CHECK: xxsldwi 0, 0, 0, 3
; CHECK: xxinsertw 34, 0, 4
  %vecins = insertelement <4 x float> %a, float %b, i32 1
  ret <4 x float> %vecins
}

define <4 x float> @_Z10testInsEltILj2EDv4_ffET0_S1_T1_(<4 x float> %a, float %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj2EDv4_ffET0_S1_T1_
; CHECK: xscvdpspn 0, 1
; CHECK: xxsldwi 0, 0, 0, 3
; CHECK: xxinsertw 34, 0, 8
  %vecins = insertelement <4 x float> %a, float %b, i32 2
  ret <4 x float> %vecins
}

define <4 x float> @_Z10testInsEltILj3EDv4_ffET0_S1_T1_(<4 x float> %a, float %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj3EDv4_ffET0_S1_T1_
; CHECK: xscvdpspn 0, 1
; CHECK: xxsldwi 0, 0, 0, 3
; CHECK: xxinsertw 34, 0, 12
  %vecins = insertelement <4 x float> %a, float %b, i32 3
  ret <4 x float> %vecins
}

define <4 x i32> @_Z10testInsEltILj0EDv4_jjET0_S1_T1_(<4 x i32> %a, i32 zeroext %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj0EDv4_jjET0_S1_T1_
; CHECK: mtfprwz 0, 3
; CHECK: xxinsertw 34, 0, 0
  %vecins = insertelement <4 x i32> %a, i32 %b, i32 0
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z10testInsEltILj1EDv4_jjET0_S1_T1_(<4 x i32> %a, i32 zeroext %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj1EDv4_jjET0_S1_T1_
; CHECK: mtfprwz 0, 3
; CHECK: xxinsertw 34, 0, 4
  %vecins = insertelement <4 x i32> %a, i32 %b, i32 1
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z10testInsEltILj2EDv4_jjET0_S1_T1_(<4 x i32> %a, i32 zeroext %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj2EDv4_jjET0_S1_T1_
; CHECK: mtfprwz 0, 3
; CHECK: xxinsertw 34, 0, 8
  %vecins = insertelement <4 x i32> %a, i32 %b, i32 2
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z10testInsEltILj3EDv4_jjET0_S1_T1_(<4 x i32> %a, i32 zeroext %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj3EDv4_jjET0_S1_T1_
; CHECK: mtfprwz 0, 3
; CHECK: xxinsertw 34, 0, 12
  %vecins = insertelement <4 x i32> %a, i32 %b, i32 3
  ret <4 x i32> %vecins
}

define <4 x float> @_Z7testInsILj0ELj0EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj0EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj1EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj1EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 0
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 1, i32 5, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj2EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj2EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 2, i32 5, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj3EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj3EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 3, i32 5, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj0EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj0EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 0, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj1EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj1EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 4
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj2EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj2EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 2, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj3EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj3EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 3, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj0EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj0EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 0, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj1EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj1EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 8
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 1, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj2EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj2EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj3EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj3EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 3, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj0EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj0EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 0>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj1EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj1EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 12
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 1>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj2EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj2EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 2>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj3EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj3EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 3>
  ret <4 x float> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj0EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj0EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj1EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj1EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 0
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 1, i32 5, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj2EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj2EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 2, i32 5, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj3EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj3EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 3, i32 5, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj0EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj0EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 0, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj1EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj1EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 4
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj2EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj2EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 2, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj3EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj3EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 3, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj0EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj0EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 0, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj1EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj1EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 8
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 1, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj2EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj2EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj3EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj3EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 3, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj0EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj0EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 0>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj1EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj1EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 12
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 1>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj2EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj2EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 2>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj3EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj3EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 3>
  ret <4 x i32> %vecins
}
define <4 x float> @testSameVecEl0BE(<4 x float> %a) {
entry:
; CHECK-LABEL: testSameVecEl0BE
; CHECK: xxinsertw 34, 34, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 5, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}
define <4 x float> @testSameVecEl2BE(<4 x float> %a) {
entry:
; CHECK-LABEL: testSameVecEl2BE
; CHECK: xxinsertw 34, 34, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x float> %vecins
}
define <4 x float> @testSameVecEl3BE(<4 x float> %a) {
entry:
; CHECK-LABEL: testSameVecEl3BE
; CHECK: xxinsertw 34, 34, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  ret <4 x float> %vecins
}
define <4 x float> @insertVarF(<4 x float> %a, float %f, i32 %el) {
entry:
; CHECK-LABEL: insertVarF
; CHECK: stfsx 1,
; CHECK: lxv
  %vecins = insertelement <4 x float> %a, float %f, i32 %el
  ret <4 x float> %vecins
}
define <4 x i32> @insertVarI(<4 x i32> %a, i32 %i, i32 %el) {
entry:
; CHECK-LABEL: insertVarI
; CHECK: stwx
; CHECK: lxv
  %vecins = insertelement <4 x i32> %a, i32 %i, i32 %el
  ret <4 x i32> %vecins
}
