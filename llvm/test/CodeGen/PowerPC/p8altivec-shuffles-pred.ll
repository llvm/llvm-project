; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-i64:64-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

; Function Attrs: nounwind
define <2 x i32> @test1(<4 x i32> %wide.vec) #0 {
entry:
  %strided.vec = shufflevector <4 x i32> %wide.vec, <4 x i32> undef, <2 x i32> <i32 0, i32 2>
  ret <2 x i32> %strided.vec

; CHECK-LABEL: @test1
; CHECK: xxswapd 35, 34
; CHECK: vmrghw 2, 2, 3
; CHECK: blr
}

; Function Attrs: nounwind
define <16 x i8> @test2(<16 x i8> %wide.vec) #0 {
entry:
  %strided.vec = shufflevector <16 x i8> %wide.vec, <16 x i8> undef, <16 x i32> <i32 undef, i32 undef, i32 undef, i32 undef, i32 0, i32 1, i32 2, i32 3, i32 undef, i32 undef, i32 undef, i32 undef, i32 8, i32 9, i32 10, i32 11>
  ret <16 x i8> %strided.vec

; CHECK-LABEL: @test2
; CHECK: vsldoi 2, 2, 2, 12
; CHECK: blr
}

attributes #0 = { nounwind "target-cpu"="pwr7" }

