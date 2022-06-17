; RUN: llc -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX11 %s
; RUN: llc -global-isel -march=amdgcn -mcpu=gfx1100 -verify-machineinstrs < %s | FileCheck -check-prefix=GFX11 %s

; GFX11-LABEL: {{^}}lds_direct_load:
; GFX11: s_mov_b32 m0
; GFX11: lds_direct_load v{{[0-9]+}}
; GFX11: s_mov_b32 m0
; GFX11: lds_direct_load v{{[0-9]+}}
; GFX11: s_mov_b32 m0
; GFX11: lds_direct_load v{{[0-9]+}}
; GFX11: v_add_f32
; GFX11: buffer_store_b32
; GFX11: buffer_store_b32
; GFX11: buffer_store_b32
; GFX11: buffer_store_b32
; GFX11: buffer_store_b32
; GFX11: buffer_store_b32
define amdgpu_ps void @lds_direct_load(<4 x i32> inreg %buf, i32 inreg %arg0,
                                       i32 inreg %arg1, i32 inreg %arg2) #0 {
main_body:
  %p0 = call float @llvm.amdgcn.lds.direct.load(i32 %arg0)
  ; Ensure memory clustering is occuring for lds_direct_load
  %p5 = fadd float %p0, 1.0
  %p1 = call float @llvm.amdgcn.lds.direct.load(i32 %arg1)
  %p2 = call float @llvm.amdgcn.lds.direct.load(i32 %arg2)
  %p3 = call float @llvm.amdgcn.lds.direct.load(i32 %arg1)
  %p4 = call float @llvm.amdgcn.lds.direct.load(i32 %arg2)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %p5, <4 x i32> %buf, i32 4, i32 0, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %p1, <4 x i32> %buf, i32 4, i32 1, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %p2, <4 x i32> %buf, i32 4, i32 2, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %p3, <4 x i32> %buf, i32 4, i32 3, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %p4, <4 x i32> %buf, i32 4, i32 4, i32 0)
  call void @llvm.amdgcn.raw.buffer.store.f32(float %p0, <4 x i32> %buf, i32 4, i32 5, i32 0)
  ret void
}

declare float @llvm.amdgcn.lds.direct.load(i32) #1
declare void @llvm.amdgcn.raw.buffer.store.f32(float, <4 x i32>, i32, i32, i32)

attributes #0 = { nounwind }
attributes #1 = { nounwind readonly }
