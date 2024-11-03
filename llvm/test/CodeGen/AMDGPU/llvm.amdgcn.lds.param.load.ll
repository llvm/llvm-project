; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX11 %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX11 %s

; GFX11-LABEL: {{^}}lds_param_load:
; GFX11: s_mov_b32 m0
; GFX11-DAG: lds_param_load v{{[0-9]+}}, attr0.x
; GFX11-DAG: lds_param_load v{{[0-9]+}}, attr0.y
; GFX11-DAG: lds_param_load v{{[0-9]+}}, attr0.z
; GFX11-DAG: lds_param_load v{{[0-9]+}}, attr0.w
; GFX11-DAG: lds_param_load v{{[0-9]+}}, attr1.x
; GFX11: s_waitcnt expcnt(4)
; GFX11: v_add_f32
; GFX11: buffer_store_b32
; GFX11: s_waitcnt expcnt(3)
; GFX11: buffer_store_b32
; GFX11: s_waitcnt expcnt(2)
; GFX11: buffer_store_b32
; GFX11: s_waitcnt expcnt(1)
; GFX11: buffer_store_b32
; GFX11: s_waitcnt expcnt(0)
; GFX11: buffer_store_b32
; GFX11: buffer_store_b32
define amdgpu_ps void @lds_param_load(ptr addrspace(8) inreg %buf, i32 inreg %arg) #0 {
main_body:
  %p0 = call float @llvm.amdgcn.lds.param.load(i32 0, i32 0, i32 %arg)
  ; Ensure memory clustering is occuring for lds_param_load
  %p5 = fadd float %p0, 1.0
  %p1 = call float @llvm.amdgcn.lds.param.load(i32 1, i32 0, i32 %arg)
  %p2 = call float @llvm.amdgcn.lds.param.load(i32 2, i32 0, i32 %arg)
  %p3 = call float @llvm.amdgcn.lds.param.load(i32 3, i32 0, i32 %arg)
  %p4 = call float @llvm.amdgcn.lds.param.load(i32 0, i32 1, i32 %arg)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %p5, ptr addrspace(8) %buf, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %p1, ptr addrspace(8) %buf, i32 4, i32 1, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %p2, ptr addrspace(8) %buf, i32 4, i32 2, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %p3, ptr addrspace(8) %buf, i32 4, i32 3, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %p4, ptr addrspace(8) %buf, i32 4, i32 4, i32 0)
  call void @llvm.amdgcn.raw.ptr.buffer.store.f32(float %p0, ptr addrspace(8) %buf, i32 4, i32 5, i32 0)
  ret void
}

declare float @llvm.amdgcn.lds.param.load(i32, i32, i32) #1
declare void @llvm.amdgcn.raw.ptr.buffer.store.f32(float, ptr addrspace(8), i32, i32, i32)

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
