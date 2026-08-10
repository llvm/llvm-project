; RUN: llvm-as < %s | llvm-dis | FileCheck %s

declare float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8), i32, i32, i32 immarg)
declare <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8), i32, i32, i32 immarg)
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8), i32, i32, i32 immarg)
declare void @llvm.amdgcn.raw.ptr.buffer.store.format.v4f32(<4 x float>, ptr addrspace(8), i32, i32, i32 immarg)
declare i32 @llvm.amdgcn.raw.ptr.atomic.buffer.load.i32(ptr addrspace(8), i32, i32, i32 immarg)
declare i32 @llvm.amdgcn.raw.ptr.buffer.atomic.add.i32(i32, ptr addrspace(8), i32, i32, i32 immarg)
declare i32 @llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.i32(i32, i32, ptr addrspace(8), i32, i32, i32 immarg)
declare float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8), i32, i32, i32, i32 immarg)
declare void @llvm.amdgcn.struct.ptr.buffer.store.f32(float, ptr addrspace(8), i32, i32, i32, i32 immarg)
declare i32 @llvm.amdgcn.struct.ptr.atomic.buffer.load.i32(ptr addrspace(8), i32, i32, i32, i32 immarg)
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.add.i32(i32, ptr addrspace(8), i32, i32, i32, i32 immarg)
declare i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32, i32, ptr addrspace(8), i32, i32, i32, i32 immarg)

define void @test(i32 %v, ptr addrspace(8) %rsrc, float %f, <4 x float> %f4) {
; CHECK-LABEL: define void @test(
; CHECK: call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %f, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call void @llvm.amdgcn.raw.ptr.buffer.store.format.v4f32(<4 x float> %f4, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call i32 @llvm.amdgcn.raw.ptr.atomic.buffer.load.i32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.add.i32(i32 %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.i32(i32 %v, i32 %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %f, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call i32 @llvm.amdgcn.struct.ptr.atomic.buffer.load.i32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.add.i32(i32 %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0, metadata !0)
; CHECK: call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %v, i32 %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0, metadata !0)
  %l = call float @llvm.amdgcn.raw.ptr.buffer.load.f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  %lf = call <4 x float> @llvm.amdgcn.raw.ptr.buffer.load.format.v4f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %f, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.format.v4f32(<4 x float> %f4, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  %a = call i32 @llvm.amdgcn.raw.ptr.atomic.buffer.load.i32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  %b = call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.add.i32(i32 %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  %c = call i32 @llvm.amdgcn.raw.ptr.buffer.atomic.cmpswap.i32(i32 %v, i32 %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  %sl = call float @llvm.amdgcn.struct.ptr.buffer.load.f32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  call void @llvm.amdgcn.struct.ptr.buffer.store.f32(float %f, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %d = call i32 @llvm.amdgcn.struct.ptr.atomic.buffer.load.i32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %e = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.add.i32(i32 %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %g = call i32 @llvm.amdgcn.struct.ptr.buffer.atomic.cmpswap.i32(i32 %v, i32 %v, ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  ret void
}

; CHECK: !0 = !{}
