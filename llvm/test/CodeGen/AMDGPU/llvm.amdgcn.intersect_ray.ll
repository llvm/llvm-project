; RUN: llc -march=amdgcn -mcpu=gfx1030 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -march=amdgcn -mcpu=gfx1013 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: not --crash llc -march=amdgcn -mcpu=gfx1012 -verify-machineinstrs < %s 2>&1 | FileCheck -check-prefix=ERR %s

; uint4 llvm.amdgcn.image.bvh.intersect.ray.i32.v4f32(uint node_ptr, float ray_extent, float4 ray_origin, float4 ray_dir, float4 ray_inv_dir, uint4 texture_descr)
; uint4 llvm.amdgcn.image.bvh.intersect.ray.i32.v4f16(uint node_ptr, float ray_extent, float4 ray_origin, half4 ray_dir, half4 ray_inv_dir, uint4 texture_descr)
; uint4 llvm.amdgcn.image.bvh.intersect.ray.i64.v4f32(ulong node_ptr, float ray_extent, float4 ray_origin, float4 ray_dir, float4 ray_inv_dir, uint4 texture_descr)
; uint4 llvm.amdgcn.image.bvh.intersect.ray.i64.v4f16(ulong node_ptr, float ray_extent, float4 ray_origin, half4 ray_dir, half4 ray_inv_dir, uint4 texture_descr)

declare <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i32.v4f32(i32, float, <4 x float>, <4 x float>, <4 x float>, <4 x i32>)
declare <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i32.v4f16(i32, float, <4 x float>, <4 x half>, <4 x half>, <4 x i32>)
declare <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i64.v4f32(i64, float, <4 x float>, <4 x float>, <4 x float>, <4 x i32>)
declare <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i64.v4f16(i64, float, <4 x float>, <4 x half>, <4 x half>, <4 x i32>)

; GCN-LABEL: {{^}}image_bvh_intersect_ray:
; GCN: image_bvh_intersect_ray v[0:3], v[0:15], s[0:3]{{$}}
; ERR: in function image_bvh_intersect_ray{{.*}}intrinsic not supported on subtarget
; Arguments are flattened to represent the actual VGPR_A layout, so we have no
; extra moves in the generated kernel.
define amdgpu_ps <4 x float> @image_bvh_intersect_ray(i32 %node_ptr, float %ray_extent, float %ray_origin_x, float %ray_origin_y, float %ray_origin_z, float %ray_dir_x, float %ray_dir_y, float %ray_dir_z, float %ray_inv_dir_x, float %ray_inv_dir_y, float %ray_inv_dir_z, <4 x i32> inreg %tdescr) {
main_body:
  %ray_origin0 = insertelement <4 x float> undef, float %ray_origin_x, i32 0
  %ray_origin1 = insertelement <4 x float> %ray_origin0, float %ray_origin_y, i32 1
  %ray_origin = insertelement <4 x float> %ray_origin1, float %ray_origin_z, i32 2
  %ray_dir0 = insertelement <4 x float> undef, float %ray_dir_x, i32 0
  %ray_dir1 = insertelement <4 x float> %ray_dir0, float %ray_dir_y, i32 1
  %ray_dir = insertelement <4 x float> %ray_dir1, float %ray_dir_z, i32 2
  %ray_inv_dir0 = insertelement <4 x float> undef, float %ray_inv_dir_x, i32 0
  %ray_inv_dir1 = insertelement <4 x float> %ray_inv_dir0, float %ray_inv_dir_y, i32 1
  %ray_inv_dir = insertelement <4 x float> %ray_inv_dir1, float %ray_inv_dir_z, i32 2
  %v = call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i32.v4f32(i32 %node_ptr, float %ray_extent, <4 x float> %ray_origin, <4 x float> %ray_dir, <4 x float> %ray_inv_dir, <4 x i32> %tdescr)
 %r = bitcast <4 x i32> %v to <4 x float>
 ret <4 x float> %r
}

; GCN-LABEL: {{^}}image_bvh_intersect_ray_a16:
; GCN: image_bvh_intersect_ray v[0:3], v[{{[0-9:]+}}], s[{{[0-9:]+}}] a16{{$}}
define amdgpu_ps <4 x float> @image_bvh_intersect_ray_a16(i32 inreg %node_ptr, float inreg %ray_extent, <4 x float> inreg %ray_origin, <4 x half> inreg %ray_dir, <4 x half> inreg %ray_inv_dir, <4 x i32> inreg %tdescr) {
main_body:
  %v = call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i32.v4f16(i32 %node_ptr, float %ray_extent, <4 x float> %ray_origin, <4 x half> %ray_dir, <4 x half> %ray_inv_dir, <4 x i32> %tdescr)
  %r = bitcast <4 x i32> %v to <4 x float>
  ret <4 x float> %r
}

; GCN-LABEL: {{^}}image_bvh64_intersect_ray:
; GCN:  image_bvh64_intersect_ray v[0:3], v[0:15], s[0:3]{{$}}
; Arguments are flattened to represent the actual VGPR_A layout, so we have no
; extra moves in the generated kernel.
define amdgpu_ps <4 x float> @image_bvh64_intersect_ray(<2 x i32> %node_ptr_vec, float %ray_extent, float %ray_origin_x, float %ray_origin_y, float %ray_origin_z, float %ray_dir_x, float %ray_dir_y, float %ray_dir_z, float %ray_inv_dir_x, float %ray_inv_dir_y, float %ray_inv_dir_z, <4 x i32> inreg %tdescr) {
main_body:
  %node_ptr = bitcast <2 x i32> %node_ptr_vec to i64
  %ray_origin0 = insertelement <4 x float> undef, float %ray_origin_x, i32 0
  %ray_origin1 = insertelement <4 x float> %ray_origin0, float %ray_origin_y, i32 1
  %ray_origin = insertelement <4 x float> %ray_origin1, float %ray_origin_z, i32 2
  %ray_dir0 = insertelement <4 x float> undef, float %ray_dir_x, i32 0
  %ray_dir1 = insertelement <4 x float> %ray_dir0, float %ray_dir_y, i32 1
  %ray_dir = insertelement <4 x float> %ray_dir1, float %ray_dir_z, i32 2
  %ray_inv_dir0 = insertelement <4 x float> undef, float %ray_inv_dir_x, i32 0
  %ray_inv_dir1 = insertelement <4 x float> %ray_inv_dir0, float %ray_inv_dir_y, i32 1
  %ray_inv_dir = insertelement <4 x float> %ray_inv_dir1, float %ray_inv_dir_z, i32 2
  %v = call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i64.v4f32(i64 %node_ptr, float %ray_extent, <4 x float> %ray_origin, <4 x float> %ray_dir, <4 x float> %ray_inv_dir, <4 x i32> %tdescr)
 %r = bitcast <4 x i32> %v to <4 x float>
 ret <4 x float> %r
}

; GCN-LABEL: {{^}}image_bvh64_intersect_ray_a16:
; GCN: image_bvh64_intersect_ray v[0:3], v[{{[0-9:]+}}], s[{{[0-9:]+}}] a16{{$}}
define amdgpu_ps <4 x float> @image_bvh64_intersect_ray_a16(i64 inreg %node_ptr, float inreg %ray_extent, <4 x float> inreg %ray_origin, <4 x half> inreg %ray_dir, <4 x half> inreg %ray_inv_dir, <4 x i32> inreg %tdescr) {
main_body:
  %v = call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i64.v4f16(i64 %node_ptr, float %ray_extent, <4 x float> %ray_origin, <4 x half> %ray_dir, <4 x half> %ray_inv_dir, <4 x i32> %tdescr)
  %r = bitcast <4 x i32> %v to <4 x float>
  ret <4 x float> %r
}

; TODO: NSA reassign is very limited and cannot work with VGPR tuples and subregs.

; GCN-LABEL: {{^}}image_bvh_intersect_ray_nsa_reassign:
; GCN: image_bvh_intersect_ray v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @image_bvh_intersect_ray_nsa_reassign(i32* %p_node_ptr, float* %p_ray, <4 x i32> inreg %tdescr) {
main_body:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep_node_ptr = getelementptr inbounds i32, i32* %p_node_ptr, i32 %lid
  %node_ptr = load i32, i32* %gep_node_ptr, align 4
  %gep_ray = getelementptr inbounds float, float* %p_ray, i32 %lid
  %ray_extent = load float, float* %gep_ray, align 4
  %ray_origin0 = insertelement <4 x float> undef, float 0.0, i32 0
  %ray_origin1 = insertelement <4 x float> %ray_origin0, float 1.0, i32 1
  %ray_origin = insertelement <4 x float> %ray_origin1, float 2.0, i32 2
  %ray_dir0 = insertelement <4 x float> undef, float 3.0, i32 0
  %ray_dir1 = insertelement <4 x float> %ray_dir0, float 4.0, i32 1
  %ray_dir = insertelement <4 x float> %ray_dir1, float 5.0, i32 2
  %ray_inv_dir0 = insertelement <4 x float> undef, float 6.0, i32 0
  %ray_inv_dir1 = insertelement <4 x float> %ray_inv_dir0, float 7.0, i32 1
  %ray_inv_dir = insertelement <4 x float> %ray_inv_dir1, float 8.0, i32 2
  %v = call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i32.v4f32(i32 %node_ptr, float %ray_extent, <4 x float> %ray_origin, <4 x float> %ray_dir, <4 x float> %ray_inv_dir, <4 x i32> %tdescr)
  store <4 x i32> %v, <4 x i32>* undef
  ret void
}

; GCN-LABEL: {{^}}image_bvh_intersect_ray_a16_nsa_reassign:
; GCN: image_bvh_intersect_ray v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] a16{{$}}
define amdgpu_kernel void @image_bvh_intersect_ray_a16_nsa_reassign(i32* %p_node_ptr, float* %p_ray, <4 x i32> inreg %tdescr) {
main_body:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep_node_ptr = getelementptr inbounds i32, i32* %p_node_ptr, i32 %lid
  %node_ptr = load i32, i32* %gep_node_ptr, align 4
  %gep_ray = getelementptr inbounds float, float* %p_ray, i32 %lid
  %ray_extent = load float, float* %gep_ray, align 4
  %ray_origin0 = insertelement <4 x float> undef, float 0.0, i32 0
  %ray_origin1 = insertelement <4 x float> %ray_origin0, float 1.0, i32 1
  %ray_origin = insertelement <4 x float> %ray_origin1, float 2.0, i32 2
  %ray_dir0 = insertelement <4 x half> undef, half 3.0, i32 0
  %ray_dir1 = insertelement <4 x half> %ray_dir0, half 4.0, i32 1
  %ray_dir = insertelement <4 x half> %ray_dir1, half 5.0, i32 2
  %ray_inv_dir0 = insertelement <4 x half> undef, half 6.0, i32 0
  %ray_inv_dir1 = insertelement <4 x half> %ray_inv_dir0, half 7.0, i32 1
  %ray_inv_dir = insertelement <4 x half> %ray_inv_dir1, half 8.0, i32 2
  %v = call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i32.v4f16(i32 %node_ptr, float %ray_extent, <4 x float> %ray_origin, <4 x half> %ray_dir, <4 x half> %ray_inv_dir, <4 x i32> %tdescr)
  store <4 x i32> %v, <4 x i32>* undef
  ret void
}

; GCN-LABEL: {{^}}image_bvh64_intersect_ray_nsa_reassign:
; GCN: image_bvh64_intersect_ray v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}]{{$}}
define amdgpu_kernel void @image_bvh64_intersect_ray_nsa_reassign(float* %p_ray, <4 x i32> inreg %tdescr) {
main_body:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep_ray = getelementptr inbounds float, float* %p_ray, i32 %lid
  %ray_extent = load float, float* %gep_ray, align 4
  %ray_origin0 = insertelement <4 x float> undef, float 0.0, i32 0
  %ray_origin1 = insertelement <4 x float> %ray_origin0, float 1.0, i32 1
  %ray_origin = insertelement <4 x float> %ray_origin1, float 2.0, i32 2
  %ray_dir0 = insertelement <4 x float> undef, float 3.0, i32 0
  %ray_dir1 = insertelement <4 x float> %ray_dir0, float 4.0, i32 1
  %ray_dir = insertelement <4 x float> %ray_dir1, float 5.0, i32 2
  %ray_inv_dir0 = insertelement <4 x float> undef, float 6.0, i32 0
  %ray_inv_dir1 = insertelement <4 x float> %ray_inv_dir0, float 7.0, i32 1
  %ray_inv_dir = insertelement <4 x float> %ray_inv_dir1, float 8.0, i32 2
  %v = call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i64.v4f32(i64 1111111111111, float %ray_extent, <4 x float> %ray_origin, <4 x float> %ray_dir, <4 x float> %ray_inv_dir, <4 x i32> %tdescr)
  store <4 x i32> %v, <4 x i32>* undef
  ret void
}

; GCN-LABEL: {{^}}image_bvh64_intersect_ray_a16_nsa_reassign:
; GCN: image_bvh64_intersect_ray v[{{[0-9:]+}}], v[{{[0-9:]+}}], s[{{[0-9:]+}}] a16{{$}}
define amdgpu_kernel void @image_bvh64_intersect_ray_a16_nsa_reassign(float* %p_ray, <4 x i32> inreg %tdescr) {
main_body:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep_ray = getelementptr inbounds float, float* %p_ray, i32 %lid
  %ray_extent = load float, float* %gep_ray, align 4
  %ray_origin0 = insertelement <4 x float> undef, float 0.0, i32 0
  %ray_origin1 = insertelement <4 x float> %ray_origin0, float 1.0, i32 1
  %ray_origin = insertelement <4 x float> %ray_origin1, float 2.0, i32 2
  %ray_dir0 = insertelement <4 x half> undef, half 3.0, i32 0
  %ray_dir1 = insertelement <4 x half> %ray_dir0, half 4.0, i32 1
  %ray_dir = insertelement <4 x half> %ray_dir1, half 5.0, i32 2
  %ray_inv_dir0 = insertelement <4 x half> undef, half 6.0, i32 0
  %ray_inv_dir1 = insertelement <4 x half> %ray_inv_dir0, half 7.0, i32 1
  %ray_inv_dir = insertelement <4 x half> %ray_inv_dir1, half 8.0, i32 2
  %v = call <4 x i32> @llvm.amdgcn.image.bvh.intersect.ray.i64.v4f16(i64 1111111111110, float %ray_extent, <4 x float> %ray_origin, <4 x half> %ray_dir, <4 x half> %ray_inv_dir, <4 x i32> %tdescr)
  store <4 x i32> %v, <4 x i32>* undef
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x()
