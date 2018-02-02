; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs< %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs< %s

; This tests for a bug that caused a crash in
; AMDGPUDAGToDAGISel::SelectMUBUFScratch() which is used for selecting
; scratch loads and stores.
; CHECK-LABEL: {{^}}store_vector_ptrs:
define amdgpu_kernel void @store_vector_ptrs(<4 x i32 addrspace(5)*> addrspace(5)* %out, <4 x [1024 x i32] addrspace(5)*> %array) nounwind {
  %p = getelementptr [1024 x i32], <4 x [1024 x i32] addrspace(5)*> %array, <4 x i16> zeroinitializer, <4 x i16> <i16 16, i16 16, i16 16, i16 16>
  store <4 x i32 addrspace(5)*> %p, <4 x i32 addrspace(5)*> addrspace(5)* %out
  ret void
}
