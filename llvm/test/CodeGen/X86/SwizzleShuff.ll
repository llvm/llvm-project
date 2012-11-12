; RUN: llc < %s -mtriple=x86_64-apple-darwin -mcpu=corei7-avx -mattr=+avx | FileCheck %s

; Check that we perform a scalar XOR on i32.

; CHECK: pull_bitcast
; CHECK: xorl
; CHECK: ret
define void @pull_bitcast (<4 x i8>* %pA, <4 x i8>* %pB) {
  %A = load <4 x i8>* %pA
  %B = load <4 x i8>* %pB
  %C = xor <4 x i8> %A, %B
  store <4 x i8> %C, <4 x i8>* %pA
  ret void
}

; CHECK: multi_use_swizzle
; CHECK: mov
; CHECK-NEXT: shuf
; CHECK-NEXT: shuf
; CHECK-NEXT: shuf
; CHECK-NEXT: xor
; CHECK-NEXT: ret
define <4 x i32> @multi_use_swizzle (<4 x i32>* %pA, <4 x i32>* %pB) {
  %A = load <4 x i32>* %pA
  %B = load <4 x i32>* %pB
  %S = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 1, i32 1, i32 5, i32 6>
  %S1 = shufflevector <4 x i32> %S, <4 x i32> undef, <4 x i32> <i32 1, i32 3, i32 2, i32 2>
  %S2 = shufflevector <4 x i32> %S, <4 x i32> undef, <4 x i32> <i32 2, i32 1, i32 0, i32 2>
  %R = xor <4 x i32> %S1, %S2
  ret <4 x i32> %R
}

; CHECK: pull_bitcast2
; CHECK: xorl
; CHECK: ret
define <4 x i8> @pull_bitcast2 (<4 x i8>* %pA, <4 x i8>* %pB, <4 x i8>* %pC) {
  %A = load <4 x i8>* %pA
  store <4 x i8> %A, <4 x i8>* %pC
  %B = load <4 x i8>* %pB
  %C = xor <4 x i8> %A, %B
  store <4 x i8> %C, <4 x i8>* %pA
  ret <4 x i8> %C
}



; CHECK: reverse_1
; CHECK-NOT: shuf
; CHECK: ret
define <4 x i32> @reverse_1 (<4 x i32>* %pA, <4 x i32>* %pB) {
  %A = load <4 x i32>* %pA
  %B = load <4 x i32>* %pB
  %S = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %S1 = shufflevector <4 x i32> %S, <4 x i32> undef, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  ret <4 x i32> %S1
}


; CHECK: no_reverse_shuff
; CHECK: shuf
; CHECK: ret
define <4 x i32> @no_reverse_shuff (<4 x i32>* %pA, <4 x i32>* %pB) {
  %A = load <4 x i32>* %pA
  %B = load <4 x i32>* %pB
  %S = shufflevector <4 x i32> %A, <4 x i32> %B, <4 x i32> <i32 1, i32 0, i32 3, i32 2>
  %S1 = shufflevector <4 x i32> %S, <4 x i32> undef, <4 x i32> <i32 3, i32 2, i32 3, i32 2>
  ret <4 x i32> %S1
}
