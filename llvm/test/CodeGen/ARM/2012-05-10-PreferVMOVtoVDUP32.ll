; RUN: llc -mtriple=arm-eabi -mcpu=swift %s -o - | FileCheck %s
; <rdar://problem/10451892>

define void @f(i32 %x, ptr %p) nounwind ssp {
entry:
; CHECK-NOT: vdup.32
  %vecinit.i = insertelement <2 x i32> undef, i32 %x, i32 0
  %vecinit1.i = insertelement <2 x i32> %vecinit.i, i32 %x, i32 1
  tail call void @llvm.arm.neon.vst1.p0.v2i32(ptr %p, <2 x i32> %vecinit1.i, i32 4)
  ret void
}

declare void @llvm.arm.neon.vst1.p0.v2i32(ptr, <2 x i32>, i32) nounwind
