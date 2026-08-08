; NOTE: This test uses intentionally invalid IR (wrong intrinsic return type)
; to ensure the auto-upgrade logic does not crash.

; RUN: opt -passes=verify -S %s 2>&1 | FileCheck %s

define <4 x ptr> @test() {
  %v = call <4 x ptr> @llvm.x86.avx512.mask.packssdw.512(<4 x i32> zeroinitializer, <4 x i32> zeroinitializer, <4 x i32> zeroinitializer)
  ret <4 x ptr> %v
}

; CHECK: Unknown intrinsic
