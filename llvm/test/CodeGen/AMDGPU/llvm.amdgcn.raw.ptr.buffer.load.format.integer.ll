; RUN: not llc -global-isel=0 -mtriple=amdgpu9.00-amd-mesa3d -filetype=null < %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=1 -mtriple=amdgpu9.00-amd-mesa3d -filetype=null < %s 2>&1 | FileCheck %s

; Only the first offending declaration is reported, so do not pin the type.
; CHECK: error: intrinsic return type (overload type 0) expected any fp or fp vector

define amdgpu_ps <4 x float> @raw_ptr_buffer_load_format_v4i32(ptr addrspace(8) inreg %rsrc) {
main_body:
  %data = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.format.v4i32(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0)
  %fdata = bitcast <4 x i32> %data to <4 x float>
  ret <4 x float> %fdata
}

define amdgpu_ps <4 x float> @raw_ptr_buffer_load_format_ofs_v4i32(ptr addrspace(8) inreg %rsrc, i32 %voffset) {
main_body:
  %data = call <4 x i32> @llvm.amdgcn.raw.ptr.buffer.load.format.v4i32(ptr addrspace(8) %rsrc, i32 %voffset, i32 0, i32 0)
  %fdata = bitcast <4 x i32> %data to <4 x float>
  ret <4 x float> %fdata
}

define amdgpu_ps float @raw_ptr_buffer_load_format_i32(ptr addrspace(8) inreg %rsrc, i32 %voffset, i32 inreg %soffset) {
main_body:
  %data = call i32 @llvm.amdgcn.raw.ptr.buffer.load.format.i32(ptr addrspace(8) %rsrc, i32 %voffset, i32 %soffset, i32 0)
  %fdata = bitcast i32 %data to float
  ret float %fdata
}
