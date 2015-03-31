;RUN: llc < %s -march=amdgcn -mcpu=verde -verify-machineinstrs | FileCheck %s
;RUN: llc < %s -march=amdgcn -mcpu=tonga -verify-machineinstrs | FileCheck %s

;CHECK: v_rcp_iflag_f32_e32

define void @test(i32 %p, i32 %q) {
   %i = udiv i32 %p, %q
   %r = bitcast i32 %i to float
   call void @llvm.SI.export(i32 15, i32 0, i32 1, i32 12, i32 0, float %r, float %r, float %r, float %r)
   ret void
}

declare void @llvm.SI.export(i32, i32, i32, i32, i32, float, float, float, float)
