; RUN: llc -march=arm -mcpu=swift < %s | FileCheck %s
; <rdar://problem/10451892>

define void @f(i32 %x, i32* %p) nounwind ssp {
entry:
; CHECK-NOT: vdup.32
  %vecinit.i = insertelement <2 x i32> undef, i32 %x, i32 0
  %vecinit1.i = insertelement <2 x i32> %vecinit.i, i32 %x, i32 1
  %0 = bitcast i32* %p to i8*
  tail call void @llvm.arm.neon.vst1.v2i32(i8* %0, <2 x i32> %vecinit1.i, i32 4)
  ret void
}

declare void @llvm.arm.neon.vst1.v2i32(i8*, <2 x i32>, i32) nounwind
