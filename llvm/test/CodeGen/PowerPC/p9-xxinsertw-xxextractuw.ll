; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64-unknown-linux-gnu \
; RUN:   -verify-machineinstrs < %s | FileCheck %s --check-prefix=CHECK-BE

define <4 x float> @_Z7testInsILj0ELj0EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj0EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj0EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj1EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj1EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj1EDv4_fET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 5, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj2EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj2EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj2EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 6, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj3EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj3EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj3EDv4_fET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 7, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj0EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj0EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj0EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj1EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj1EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj1EDv4_fET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 4
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj2EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj2EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj2EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj3EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj3EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj3EDv4_fET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 7, i32 2, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj0EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj0EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj0EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj1EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj1EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj1EDv4_fET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj2EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj2EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj2EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj3EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj3EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj3EDv4_fET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 7, i32 3>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj0EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj0EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj0EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj1EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj1EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj1EDv4_fET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj2EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj2EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj2EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj3EDv4_fET1_S1_S1_(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj3EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj3EDv4_fET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x float> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj0EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj0EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj0EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 4, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj1EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj1EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj1EDv4_jET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 0
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 5, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj2EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj2EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj2EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 6, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj3EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj3EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj3EDv4_jET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 7, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj0EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj0EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj0EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 4, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj1EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj1EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj1EDv4_jET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 4
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 5, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj2EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj2EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj2EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj3EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj3EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj3EDv4_jET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 7, i32 2, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj0EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj0EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj0EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 4, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj1EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj1EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj1EDv4_jET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 8
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj2EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj2EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj2EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 6, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj3EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj3EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj3EDv4_jET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 7, i32 3>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj0EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj0EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj0EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 4>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj1EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj1EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj1EDv4_jET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 12
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj2EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj2EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj2EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj3EDv4_jET1_S1_S1_(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj3EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj3EDv4_jET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %a, <4 x i32> %b, <4 x i32> <i32 0, i32 1, i32 2, i32 7>
  ret <4 x i32> %vecins
}

define float @_Z13testUiToFpExtILj0EEfDv4_j(<4 x i32> %a) {
entry:
; CHECK-LABEL: _Z13testUiToFpExtILj0EEfDv4_j
; CHECK: xxextractuw 0, 34, 12
; CHECK: xscvuxdsp 1, 0
; CHECK-BE-LABEL: _Z13testUiToFpExtILj0EEfDv4_j
; CHECK-BE: xxextractuw 0, 34, 0
; CHECK-BE: xscvuxdsp 1, 0
  %vecext = extractelement <4 x i32> %a, i32 0
  %conv = uitofp i32 %vecext to float
  ret float %conv
}

define float @_Z13testUiToFpExtILj1EEfDv4_j(<4 x i32> %a) {
entry:
; CHECK-LABEL: _Z13testUiToFpExtILj1EEfDv4_j
; CHECK: xxextractuw 0, 34, 8
; CHECK: xscvuxdsp 1, 0
; CHECK-BE-LABEL: _Z13testUiToFpExtILj1EEfDv4_j
; CHECK-BE: xxextractuw 0, 34, 4
; CHECK-BE: xscvuxdsp 1, 0
  %vecext = extractelement <4 x i32> %a, i32 1
  %conv = uitofp i32 %vecext to float
  ret float %conv
}

define float @_Z13testUiToFpExtILj2EEfDv4_j(<4 x i32> %a) {
entry:
; CHECK-LABEL: _Z13testUiToFpExtILj2EEfDv4_j
; CHECK: xxextractuw 0, 34, 4
; CHECK: xscvuxdsp 1, 0
; CHECK-BE-LABEL: _Z13testUiToFpExtILj2EEfDv4_j
; CHECK-BE: xxextractuw 0, 34, 8
; CHECK-BE: xscvuxdsp 1, 0
  %vecext = extractelement <4 x i32> %a, i32 2
  %conv = uitofp i32 %vecext to float
  ret float %conv
}

define float @_Z13testUiToFpExtILj3EEfDv4_j(<4 x i32> %a) {
entry:
; CHECK-LABEL: _Z13testUiToFpExtILj3EEfDv4_j
; CHECK: xxextractuw 0, 34, 0
; CHECK: xscvuxdsp 1, 0
; CHECK-BE-LABEL: _Z13testUiToFpExtILj3EEfDv4_j
; CHECK-BE: xxextractuw 0, 34, 12
; CHECK-BE: xscvuxdsp 1, 0
  %vecext = extractelement <4 x i32> %a, i32 3
  %conv = uitofp i32 %vecext to float
  ret float %conv
}

; Verify we generate optimal code for unsigned vector int elem extract followed
; by conversion to double

define double @conv2dlbTestui0(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2dlbTestui0
; CHECK: xxextractuw [[SW:[0-9]+]], 34, 12
; CHECK: xscvuxddp 1, [[SW]]
; CHECK-BE-LABEL: conv2dlbTestui0
; CHECK-BE: xxextractuw [[CP:[0-9]+]], 34, 0
; CHECK-BE: xscvuxddp 1, [[CP]]
  %0 = extractelement <4 x i32> %a, i32 0
  %1 = uitofp i32 %0 to double
  ret double %1
}

define double @conv2dlbTestui1(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2dlbTestui1
; CHECK: xxextractuw [[SW:[0-9]+]], 34, 8
; CHECK: xscvuxddp 1, [[SW]]
; CHECK-BE-LABEL: conv2dlbTestui1
; CHECK-BE: xxextractuw [[CP:[0-9]+]], 34, 4
; CHECK-BE: xscvuxddp 1, [[CP]]
  %0 = extractelement <4 x i32> %a, i32 1
  %1 = uitofp i32 %0 to double
  ret double %1
}

define double @conv2dlbTestui2(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2dlbTestui2
; CHECK: xxextractuw [[SW:[0-9]+]], 34, 4
; CHECK: xscvuxddp 1, [[SW]]
; CHECK-BE-LABEL: conv2dlbTestui2
; CHECK-BE: xxextractuw [[CP:[0-9]+]], 34, 8
; CHECK-BE: xscvuxddp 1, [[CP]]
  %0 = extractelement <4 x i32> %a, i32 2
  %1 = uitofp i32 %0 to double
  ret double %1
}

define double @conv2dlbTestui3(<4 x i32> %a) {
entry:
; CHECK-LABEL: conv2dlbTestui3
; CHECK: xxextractuw [[SW:[0-9]+]], 34, 0
; CHECK: xscvuxddp 1, [[SW]]
; CHECK-BE-LABEL: conv2dlbTestui3
; CHECK-BE: xxextractuw [[CP:[0-9]+]], 34, 12
; CHECK-BE: xscvuxddp 1, [[CP]]
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
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z10testInsEltILj0EDv4_ffET0_S1_T1_
; CHECK-BE: xscvdpspn 0, 1
; CHECK-BE: xxsldwi 0, 0, 0, 3
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = insertelement <4 x float> %a, float %b, i32 0
  ret <4 x float> %vecins
}

define <4 x float> @_Z10testInsEltILj1EDv4_ffET0_S1_T1_(<4 x float> %a, float %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj1EDv4_ffET0_S1_T1_
; CHECK: xscvdpspn 0, 1
; CHECK: xxsldwi 0, 0, 0, 3
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z10testInsEltILj1EDv4_ffET0_S1_T1_
; CHECK-BE: xscvdpspn 0, 1
; CHECK-BE: xxsldwi 0, 0, 0, 3
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = insertelement <4 x float> %a, float %b, i32 1
  ret <4 x float> %vecins
}

define <4 x float> @_Z10testInsEltILj2EDv4_ffET0_S1_T1_(<4 x float> %a, float %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj2EDv4_ffET0_S1_T1_
; CHECK: xscvdpspn 0, 1
; CHECK: xxsldwi 0, 0, 0, 3
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z10testInsEltILj2EDv4_ffET0_S1_T1_
; CHECK-BE: xscvdpspn 0, 1
; CHECK-BE: xxsldwi 0, 0, 0, 3
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = insertelement <4 x float> %a, float %b, i32 2
  ret <4 x float> %vecins
}

define <4 x float> @_Z10testInsEltILj3EDv4_ffET0_S1_T1_(<4 x float> %a, float %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj3EDv4_ffET0_S1_T1_
; CHECK: xscvdpspn 0, 1
; CHECK: xxsldwi 0, 0, 0, 3
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z10testInsEltILj3EDv4_ffET0_S1_T1_
; CHECK-BE: xscvdpspn 0, 1
; CHECK-BE: xxsldwi 0, 0, 0, 3
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = insertelement <4 x float> %a, float %b, i32 3
  ret <4 x float> %vecins
}

define <4 x i32> @_Z10testInsEltILj0EDv4_jjET0_S1_T1_(<4 x i32> %a, i32 zeroext %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj0EDv4_jjET0_S1_T1_
; CHECK: mtfprwz 0, 5
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z10testInsEltILj0EDv4_jjET0_S1_T1_
; CHECK-BE: mtfprwz 0, 5
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = insertelement <4 x i32> %a, i32 %b, i32 0
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z10testInsEltILj1EDv4_jjET0_S1_T1_(<4 x i32> %a, i32 zeroext %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj1EDv4_jjET0_S1_T1_
; CHECK: mtfprwz 0, 5
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z10testInsEltILj1EDv4_jjET0_S1_T1_
; CHECK-BE: mtfprwz 0, 5
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = insertelement <4 x i32> %a, i32 %b, i32 1
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z10testInsEltILj2EDv4_jjET0_S1_T1_(<4 x i32> %a, i32 zeroext %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj2EDv4_jjET0_S1_T1_
; CHECK: mtfprwz 0, 5
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z10testInsEltILj2EDv4_jjET0_S1_T1_
; CHECK-BE: mtfprwz 0, 5
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = insertelement <4 x i32> %a, i32 %b, i32 2
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z10testInsEltILj3EDv4_jjET0_S1_T1_(<4 x i32> %a, i32 zeroext %b) {
entry:
; CHECK-LABEL: _Z10testInsEltILj3EDv4_jjET0_S1_T1_
; CHECK: mtfprwz 0, 5
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z10testInsEltILj3EDv4_jjET0_S1_T1_
; CHECK-BE: mtfprwz 0, 5
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = insertelement <4 x i32> %a, i32 %b, i32 3
  ret <4 x i32> %vecins
}

define <4 x float> @_Z7testInsILj0ELj0EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj0EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj0EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj1EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj1EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj1EDv4_fET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 0
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 1, i32 5, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj2EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj2EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj2EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 2, i32 5, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj0ELj3EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj3EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj3EDv4_fET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 3, i32 5, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj0EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj0EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj0EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 0, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj1EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj1EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj1EDv4_fET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 4
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj2EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj2EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj2EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 2, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj1ELj3EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj3EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj3EDv4_fET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 3, i32 6, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj0EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj0EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj0EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 0, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj1EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj1EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj1EDv4_fET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 8
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 1, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj2EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj2EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj2EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj2ELj3EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj3EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj3EDv4_fET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 3, i32 7>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj0EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj0EDv4_fET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj0EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 0>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj1EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj1EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj1EDv4_fET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 12
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 1>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj2EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj2EDv4_fET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj2EDv4_fET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 2>
  ret <4 x float> %vecins
}

define <4 x float> @_Z7testInsILj3ELj3EDv4_fET1_S1_S1_r(<4 x float> %a, <4 x float> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj3EDv4_fET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj3EDv4_fET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x float> %b, <4 x float> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 3>
  ret <4 x float> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj0EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj0EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj0EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 0, i32 5, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj1EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj1EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj1EDv4_jET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 0
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 1, i32 5, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj2EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj2EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj2EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 2, i32 5, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj0ELj3EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj0ELj3EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 12
; CHECK-BE-LABEL: _Z7testInsILj0ELj3EDv4_jET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 0
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 3, i32 5, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj0EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj0EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj0EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 0, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj1EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj1EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj1EDv4_jET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 4
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 1, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj2EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj2EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj2EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 2, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj1ELj3EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj1ELj3EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 8
; CHECK-BE-LABEL: _Z7testInsILj1ELj3EDv4_jET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 4
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 3, i32 6, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj0EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj0EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj0EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 0, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj1EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj1EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj1EDv4_jET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 8
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 1, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj2EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj2EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj2EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 2, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj2ELj3EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj2ELj3EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 4
; CHECK-BE-LABEL: _Z7testInsILj2ELj3EDv4_jET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 8
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 3, i32 7>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj0EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj0EDv4_jET1_S1_S1_
; CHECK: xxswapd 0, 35
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj0EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 3
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 0>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj1EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj1EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 1
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj1EDv4_jET1_S1_S1_
; CHECK-BE-NOT: xxsldwi
; CHECK-BE: xxinsertw 34, 35, 12
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 1>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj2EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj2EDv4_jET1_S1_S1_
; CHECK-NOT: xxsldwi
; CHECK: xxinsertw 34, 35, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj2EDv4_jET1_S1_S1_
; CHECK-BE: xxsldwi 0, 35, 35, 1
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 2>
  ret <4 x i32> %vecins
}

define <4 x i32> @_Z7testInsILj3ELj3EDv4_jET1_S1_S1_r(<4 x i32> %a, <4 x i32> %b) {
entry:
; CHECK-LABEL: _Z7testInsILj3ELj3EDv4_jET1_S1_S1_
; CHECK: xxsldwi 0, 35, 35, 3
; CHECK: xxinsertw 34, 0, 0
; CHECK-BE-LABEL: _Z7testInsILj3ELj3EDv4_jET1_S1_S1_
; CHECK-BE: xxswapd 0, 35
; CHECK-BE: xxinsertw 34, 0, 12
  %vecins = shufflevector <4 x i32> %b, <4 x i32> %a, <4 x i32> <i32 4, i32 5, i32 6, i32 3>
  ret <4 x i32> %vecins
}
define <4 x float> @testSameVecEl0BE(<4 x float> %a) {
entry:
; CHECK-BE-LABEL: testSameVecEl0BE
; CHECK-BE: xxinsertw 34, 34, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 5, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}
define <4 x float> @testSameVecEl2BE(<4 x float> %a) {
entry:
; CHECK-BE-LABEL: testSameVecEl2BE
; CHECK-BE: xxinsertw 34, 34, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 5, i32 3>
  ret <4 x float> %vecins
}
define <4 x float> @testSameVecEl3BE(<4 x float> %a) {
entry:
; CHECK-BE-LABEL: testSameVecEl3BE
; CHECK-BE: xxinsertw 34, 34, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 5>
  ret <4 x float> %vecins
}
define <4 x float> @testSameVecEl0LE(<4 x float> %a) {
entry:
; CHECK-LABEL: testSameVecEl0LE
; CHECK: xxinsertw 34, 34, 12
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 6, i32 1, i32 2, i32 3>
  ret <4 x float> %vecins
}
define <4 x float> @testSameVecEl1LE(<4 x float> %a) {
entry:
; CHECK-LABEL: testSameVecEl1LE
; CHECK: xxinsertw 34, 34, 8
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 0, i32 6, i32 2, i32 3>
  ret <4 x float> %vecins
}
define <4 x float> @testSameVecEl3LE(<4 x float> %a) {
entry:
; CHECK-LABEL: testSameVecEl3LE
; CHECK: xxinsertw 34, 34, 0
  %vecins = shufflevector <4 x float> %a, <4 x float> %a, <4 x i32> <i32 0, i32 1, i32 2, i32 6>
  ret <4 x float> %vecins
}
define <4 x float> @insertVarF(<4 x float> %a, float %f, i32 %el) {
entry:
; CHECK-LABEL: insertVarF
; CHECK: stfsx 1,
; CHECK: lxv
; CHECK-BE-LABEL: insertVarF
; CHECK-BE: stfsx 1,
; CHECK-BE: lxv
  %vecins = insertelement <4 x float> %a, float %f, i32 %el
  ret <4 x float> %vecins
}
define <4 x i32> @insertVarI(<4 x i32> %a, i32 %i, i32 %el) {
entry:
; CHECK-LABEL: insertVarI
; CHECK: stwx
; CHECK: lxv
; CHECK-BE-LABEL: insertVarI
; CHECK-BE: stwx
; CHECK-BE: lxv
  %vecins = insertelement <4 x i32> %a, i32 %i, i32 %el
  ret <4 x i32> %vecins
}
define <4 x i32> @intrinsicInsertTest(<4 x i32> %a, <2 x i64> %b) {
entry:
; CHECK-LABEL:intrinsicInsertTest
; CHECK: xxinsertw 34, 35, 3
; CHECK: blr
  %ans = tail call <4 x i32> @llvm.ppc.vsx.xxinsertw(<4 x i32> %a, <2 x i64> %b, i32 3)
  ret <4 x i32> %ans
}
declare <4 x i32> @llvm.ppc.vsx.xxinsertw(<4 x i32>, <2 x i64>, i32)
define <2 x i64> @intrinsicExtractTest(<2 x i64> %a) {
entry:
; CHECK-LABEL: intrinsicExtractTest
; CHECK: xxextractuw 0, 34, 5
; CHECK: blr
  %ans = tail call <2 x i64> @llvm.ppc.vsx.xxextractuw(<2 x i64> %a, i32 5)
  ret <2 x i64> %ans
}
declare <2 x i64>  @llvm.ppc.vsx.xxextractuw(<2 x i64>, i32)
