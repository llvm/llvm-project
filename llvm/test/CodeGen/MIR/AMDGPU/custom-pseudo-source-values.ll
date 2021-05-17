; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=gfx1010 -stop-after finalize-isel -o %t.mir %s
; RUN: llc -run-pass=none -verify-machineinstrs %t.mir -o - | FileCheck %s

; Test that custom pseudo source values can be round trip serialized through MIR.

; CHECK-LABEL: {{^}}name: shader
; CHECK: %[[#]]:vgpr_32 = BUFFER_LOAD_DWORD_OFFSET killed %17, %18, 4, 0, 0, 0, implicit $exec :: (dereferenceable load 4 from custom "BufferResource" + 4, align 1, addrspace 4)
; CHECK: IMAGE_STORE_V4_V3_nsa_gfx10 killed %[[#]], %[[#]], %[[#]], %[[#]], killed %[[#]], 15, 2, -1, 0, 0, 0, 0, 0, 0, implicit $exec :: (dereferenceable store 16 into custom "ImageResource")
; CHECK: DS_GWS_BARRIER %[[#]], 63, implicit $m0, implicit $exec :: (load 4 from custom "GWSResource")
define amdgpu_cs void @shader(i32 %arg0, i32 %arg1, <8 x i32> inreg %arg2, <4 x i32> inreg %arg3) {
  %bload0 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %arg3, i32 4, i32 0, i32 0)
  %bload1 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %arg3, i32 8, i32 0, i32 0)
  %bload2 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %arg3, i32 12, i32 0, i32 0)
  %bload3 = call i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32> %arg3, i32 16, i32 0, i32 0)
  %bload0.f = bitcast i32 %bload0 to float
  %bload1.f = bitcast i32 %bload1 to float
  %bload2.f = bitcast i32 %bload2 to float
  %bload3.f = bitcast i32 %bload3 to float
  %istore0 = insertelement <4 x float> undef, float %bload0.f, i32 0
  %istore1 = insertelement <4 x float> %istore0, float %bload0.f, i32 1
  %istore2 = insertelement <4 x float> %istore1, float %bload0.f, i32 2
  %istore3 = insertelement <4 x float> %istore2, float %bload0.f, i32 3
  call void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float> %istore3, i32 15, i32 %arg0, i32 %arg1, i32 0, <8 x i32> %arg2, i32 0, i32 0)
  call void @llvm.amdgcn.ds.gws.barrier(i32 %bload0, i32 63)
  ret void
}

declare void @llvm.amdgcn.image.store.3d.v4f32.i32(<4 x float>, i32 immarg, i32, i32, i32, <8 x i32>, i32 immarg, i32 immarg) #0
declare i32 @llvm.amdgcn.raw.buffer.load.i32(<4 x i32>, i32, i32, i32 immarg) #1
declare void @llvm.amdgcn.ds.gws.barrier(i32, i32) #2

attributes #0 = { nounwind willreturn writeonly }
attributes #1 = { nounwind readonly willreturn }
attributes #2 = { convergent inaccessiblememonly nounwind }
