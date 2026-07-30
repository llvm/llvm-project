; RUN: llc < %s -float-abi=hard -debug-only arm-isel 2>&1 | FileCheck %s
; RUN: llc < %s -float-abi=soft -debug-only arm-isel 2>&1 | FileCheck %s --check-prefix=SOFT
; REQUIRES: asserts

target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-n32-S64"
target triple = "armv7-none--eabi"

; SOFT-NOT: isHA

; CHECK: isHA: 1 { float }
define void @f0b({ float } %a) {
  ret void
}

; CHECK: isHA: 1 { float, float }
define void @f1({ float, float } %a) {
  ret void
}

; CHECK: isHA: 1 { float, float, float }
define void @f1b({ float, float, float } %a) {
  ret void
}

; CHECK: isHA: 1 { float, float, float, float }
define void @f1c({ float, float, float, float } %a) {
  ret void
}

; CHECK: isHA: 0 { float, float, float, float, float }
define void @f2({ float, float, float, float, float } %a) {
  ret void
}

; CHECK: isHA: 1 { double }
define void @f3({ double } %a) {
  ret void
}

; CHECK: isHA: 1 { double, double, double, double }
define void @f4({ double, double, double, double } %a) {
  ret void
}

; CHECK: isHA: 0 { double, double, double, double, double }
define void @f5({ double, double, double, double, double } %a) {
  ret void
}

; CHECK: isHA: 0 { i32, i32 }
define void @f5b({ i32, i32 } %a) {
  ret void
}

; CHECK: isHA: 1 { [1 x float] }
define void @f6({ [1 x float] } %a) {
  ret void
}

; CHECK: isHA: 1 { [4 x float] }
define void @f7({ [4 x float] } %a) {
  ret void
}

; CHECK: isHA: 0 { [5 x float] }
define void @f8({ [5 x float] } %a) {
  ret void
}

; CHECK: isHA: 1 [1 x float]
define void @f6b([1 x float] %a) {
  ret void
}

; CHECK: isHA: 1 [4 x float]
define void @f7b([4 x float] %a) {
  ret void
}

; CHECK: isHA: 0 [5 x float]
define void @f8b([5 x float] %a) {
  ret void
}

; CHECK: isHA: 1 { [2 x float], [2 x float] }
define void @f9({ [2 x float], [2 x float] } %a) {
  ret void
}

; CHECK: isHA: 1 { [1 x float], [3 x float] }
define void @f9b({ [1 x float], [3 x float] } %a) {
  ret void
}

; CHECK: isHA: 0 { [3 x float], [3 x float] }
define void @f10({ [3 x float], [3 x float] } %a) {
  ret void
}

; CHECK: isHA: 1 { <2 x float> }
define void @f11({ <2 x float>  } %a) {
  ret void
}

; CHECK: isHA: 0 { <3 x float> }
define void @f12({ <3 x float>  } %a) {
  ret void
}

; CHECK: isHA: 1 { <4 x float> }
define void @f13({ <4 x float>  } %a) {
  ret void
}

; CHECK: isHA: 1 { <2 x float>, <2 x float> }
define void @f15({ <2 x float>, <2 x float>  } %a) {
  ret void
}

; CHECK: isHA: 0 { <2 x float>, float }
define void @f15b({ <2 x float>, float  } %a) {
  ret void
}

; CHECK: isHA: 0 { <2 x float>, [2 x float] }
define void @f15c({ <2 x float>, [2 x float]  } %a) {
  ret void
}

; CHECK: isHA: 0 { <2 x float>, <4 x float> }
define void @f16({ <2 x float>, <4 x float>  } %a) {
  ret void
}

; CHECK: isHA: 1 { <2 x double> }
define void @f17({ <2 x double>  } %a) {
  ret void
}

; CHECK: isHA: 1 { <2 x i32> }
define void @f18({ <2 x i32>  } %a) {
  ret void
}

; CHECK: isHA: 1 { <2 x i64>, <4 x i32> }
define void @f19({ <2 x i64>, <4 x i32> } %a) {
  ret void
}

; CHECK: isHA: 1 { [4 x <4 x float>] }
define void @f20({ [4 x <4 x float>]  } %a) {
  ret void
}

; CHECK: isHA: 0 { [5 x <4 x float>] }
define void @f21({ [5 x <4 x float>]  } %a) {
  ret void
}

; CHECK-NOT: isHA
define void @f22({ float } %a, ...) {
  ret void
}

