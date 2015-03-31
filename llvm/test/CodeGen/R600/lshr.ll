;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK: s_lshr_b32 s{{[0-9]}}, s{{[0-9]}}, 1

define void @test(i32 %p) {
   %i = udiv i32 %p, 2
   %r = bitcast i32 %i to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %r, float %r, float %r, float %r)
   ret void
}

declare <4 x float> @llvm.SI.sample.(i32, <4 x i32>, <8 x i32>, <4 x i32>, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
