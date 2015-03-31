;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK: v_mov_b32_e32 v{{[0-9]+}}, 0xaaaaaaab
;CHECK: v_mul_hi_u32 v0, {{v[0-9]+}}, {{s[0-9]+}}
;CHECK-NEXT: v_lshrrev_b32_e32 v0, 1, v0

define void @test(i32 %p) {
   %i = udiv i32 %p, 3
   %r = bitcast i32 %i to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %r, float %r, float %r, float %r)
   ret void
}

declare <4 x float> @llvm.SI.sample.(i32, <4 x i32>, <8 x i32>, <4 x i32>, i32) readnone

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
