; RUN: llc -verify-machineinstrs -ppc-disable-perfect-shuffle=false < %s | FileCheck %s

; TODO: Fix this case when disabling perfect shuffle

target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define <2 x i32> @test1(<4 x i32> %wide.vec) #0 {
entry:
  %strided.vec = shufflevector <4 x i32> %wide.vec, <4 x i32> undef, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %strided.vec

; CHECK-LABEL: @test1
; CHECK: xxswapd 0, 34
; CHECK: xxmrghw 34, 34, 0
; CHECK: blr
}

; Function Attrs: nounwind
define <16 x i8> @test2(<16 x i8> %wide.vec) #0 {
entry:
  %strided.vec = shufflevector <16 x i8> %wide.vec, <16 x i8> undef, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 8, i32 9, i32 10, i32 11>
  ret <16 x i8> %strided.vec

; CHECK-LABEL: @test2
; CHECK: xxsldwi 34, 34, 34, 3
; CHECK: blr
}

attributes #0 = { nounwind "target-cpu"="pwr7" }

