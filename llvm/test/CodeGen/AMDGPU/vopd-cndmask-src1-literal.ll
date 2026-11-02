; Keep two masked buffer loads live. Both selects use the same
; predicate and the same non-inline src1 immediate.
;
; RUN: llc -mtriple=amdgpu11.00 -verify-machineinstrs %s -o - | FileCheck -check-prefixes=GCN,GFX11 %s
; RUN: llc -mtriple=amdgpu11.70 -verify-machineinstrs %s -o - | FileCheck -check-prefixes=GCN,GFX11 %s
; RUN: llc -mtriple=amdgpu12.00 -verify-machineinstrs %s -o - | FileCheck -check-prefixes=GCN,GFX12 %s
; RUN: llc -mtriple=amdgpu12.50 -verify-machineinstrs %s -o - | FileCheck -check-prefixes=GCN,GFX12 %s
; RUN: llc -mtriple=amdgpu13.10 -verify-machineinstrs %s -o - | FileCheck -check-prefixes=GCN,GFX12 %s
; RUN: llc -mtriple=amdgpu12.50 -amdgpu-enable-vopd=0 -verify-machineinstrs %s -o - | FileCheck -check-prefixes=GCN,DISABLED %s

source_filename = "attn_fwd_kernel.simplified.llir"
target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-p10:32:32-p11:32:32-p12:32:32-p13:32:32-p14:32:32-p15:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgpu12.50-amd-amdhsa"

declare i32 @llvm.amdgcn.workgroup.id.x()
declare i32 @llvm.amdgcn.workgroup.id.y()
declare i32 @llvm.amdgcn.workgroup.id.z()
declare i32 @llvm.amdgcn.workitem.id.x()
declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1), i16, i64, i32)
declare <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8), i32, i32, i32 immarg)

define amdgpu_kernel void @attn_fwd_kernel_simplified(
    ptr addrspace(1) readonly %input, ptr addrspace(1) writeonly %output,
    i32 %batch_stride, i32 %head_stride, i32 %row_stride) #0 !reqd_work_group_size !0 {
; GCN-LABEL: attn_fwd_kernel_simplified:
; GFX11-NOT: v_mov_b32_e32 {{v[0-9]+}}, 0x3039
; GFX11-NOT: v_dual_cndmask_b32
; GFX11: v_cndmask_b32_e64 v[[ADDR32:[0-9]+]], {{v[0-9]+}}, 0x3039, {{(s[0-9]+|vcc_lo)}}
; GFX11-NEXT: v_cndmask_b32_e64 v[[ADDR80:[0-9]+]], {{v[0-9]+}}, 0x3039, {{(s[0-9]+|vcc_lo)}}
; GFX11-NOT: v_dual_cndmask_b32
; GFX11: s_endpgm
; GFX12: v_mov_b32_e32 [[MASK:v[0-9]+]], 0x3039
; GFX12-NOT: v_cndmask_b32
; GFX12: v_dual_cndmask_b32 v[[ADDR32:[0-9]+]], {{v[0-9]+}}, [[MASK]]{{(, vcc_lo)?}} :: v_dual_cndmask_b32 v[[ADDR80:[0-9]+]], {{v[0-9]+}}, [[MASK]]{{(, vcc_lo)?}}
; GFX12-NOT: v_cndmask_b32
; GFX12: s_endpgm
; DISABLED-NOT: v_mov_b32_e32 {{v[0-9]+}}, 0x3039
; DISABLED-NOT: v_dual_cndmask_b32
; DISABLED: v_cndmask_b32_e64 v[[ADDR32:[0-9]+]], {{v[0-9]+}}, 0x3039, {{(s[0-9]+|vcc_lo)}}
; DISABLED: v_cndmask_b32_e64 v[[ADDR80:[0-9]+]], {{v[0-9]+}}, 0x3039, {{(s[0-9]+|vcc_lo)}}
; DISABLED-NOT: v_dual_cndmask_b32
; DISABLED: s_endpgm
entry:
  %block_x = call i32 @llvm.amdgcn.workgroup.id.x()
  %block_y = call i32 @llvm.amdgcn.workgroup.id.y()
  %block_z = call i32 @llvm.amdgcn.workgroup.id.z()
  %block_row = shl i32 %block_z, 8
  %thread = call i32 @llvm.amdgcn.workitem.id.x()
  %lane = and i32 %thread, 31
  %wave = lshr i32 %thread, 5
  %wave_half = lshr i32 %wave, 1
  %wave_bit = and i32 %wave_half, 1
  %wave_order = xor i32 %wave_bit, %wave
  %wave_row = shl i32 %wave_order, 6
  %wave_row_bits = and i32 %wave_row, 192
  %local_row = or disjoint i32 %wave_row_bits, %lane
  %row = or disjoint i32 %local_row, %block_row
  %invalid = icmp sge i32 %row, 256
  %batch_origin = mul i32 %batch_stride, %block_x
  %head_origin = mul i32 %head_stride, %block_y
  %origin = add i32 %head_origin, %batch_origin
  %row_origin = mul i32 %row_stride, %row
  %element_offset = add i32 %origin, %row_origin
  %resource = call ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1.i64(ptr addrspace(1) %input, i16 0, i64 2147483646, i32 159744)
  %base = shl i32 %element_offset, 1
  %offset32 = add i32 %base, 32
  %masked32 = select i1 %invalid, i32 12345, i32 %offset32
  %value32 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %resource, i32 %masked32, i32 0, i32 0)
  %offset80 = add i32 %base, 80
  %masked80 = select i1 %invalid, i32 12345, i32 %offset80
  %value80 = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.v4i32(ptr addrspace(8) %resource, i32 %masked80, i32 0, i32 0)
  %output_index = shl i32 %thread, 1
  %output32 = getelementptr <4 x i32>, ptr addrspace(1) %output, i32 %output_index
  %output80 = getelementptr <4 x i32>, ptr addrspace(1) %output32, i32 1
  store <4 x i32> %value32, ptr addrspace(1) %output32, align 16
  store <4 x i32> %value80, ptr addrspace(1) %output80, align 16
  ret void
}

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="896,896" "amdgpu-waves-per-eu"="7,7" "uniform-work-group-size"="true" }

!0 = !{i32 896, i32 1, i32 1}
