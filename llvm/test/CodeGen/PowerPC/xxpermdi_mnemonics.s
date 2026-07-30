; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 --vec-extabi < %s | FileCheck --check-prefixes=CHECK,OLD %s
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr8 --vec-extabi < %s | FileCheck --check-prefixes=CHECK,OLD %s

; RUN: llc -mtriple powerpc64-unknown-linux-gnu -mcpu=pwr8 < %s | FileCheck --check-prefixes=CHECK,MODERN  %s
; RUN: llc -mtriple powerpc64-ibm-aix-xcoff -mcpu=pwr8 -mattr=+modern-aix-as --vec-extabi < %s | FileCheck --check-prefixes=CHECK,MODERN %s
; RUN: llc -mtriple powerpc-ibm-aix-xcoff -mcpu=pwr8 -mattr=+modern-aix-as --vec-extabi < %s | FileCheck --check-prefixes=CHECK,MODERN %s

define <2 x double> @splat1(<2 x double> %A, <2 x double> %B) {
entry:
  %0 = shufflevector <2 x double> %B, <2 x double> undef, <2 x i32> <i32 0, i32 0>
  ret <2 x double> %0
}

; CHECK-LABEL: splat1
; OLD:    xxmrghd 34, 35, 35
; MODERN: xxspltd 34, 35, 0

define <2 x double> @splat2(<2 x double> %A, <2 x double> %B) {
entry:
  %0 = shufflevector <2 x double> %B, <2 x double> undef, <2 x i32> <i32 1, i32 1>
  ret <2 x double> %0
}

; CHECK-LABEL: splat2
; OLD:    xxmrgld 34, 35, 35
; MODERN: xxspltd 34, 35, 1

define <2 x double> @swap(<2 x double> %A, <2 x double> %B) {
entry:
  %0 = shufflevector <2 x double> %B, <2 x double> undef, <2 x i32> <i32 1, i32 0>
  ret <2 x double> %0
}

; CHECK-LABEL: swap
; CHECK: xxswapd 34, 35

define <2 x double> @mergehi(<2 x double> %A, <2 x double> %B) {
entry:
  %0 = shufflevector <2 x double> %A, <2 x double> %B, <2 x i32> <i32 0, i32 2>
  ret <2 x double> %0
}

; CHECK-LABEL: mergehi
; CHECK: xxmrghd 34, 34, 35

define <2 x double> @mergelo(<2 x double> %A, <2 x double> %B) {
entry:
  %0 = shufflevector <2 x double> %A, <2 x double> %B, <2 x i32> <i32 1, i32 3>
  ret <2 x double> %0
}

; CHECK-LABEL: mergelo
; CHECK: xxmrgld 34, 34, 35
