; RUN: not llc -global-isel=0 -mtriple=amdgpu9.00 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel -mtriple=amdgpu9.00 -filetype=null %s 2>&1 | FileCheck %s

; An i8 buffer.load.format / buffer.store.format has no corresponding real
; instruction (no byte-granularity format access exists in hardware), so both
; SelectionDAG and GlobalISel must refuse to lower it.

; CHECK: error: {{.*}}unsupported sub-dword format buffer load
define amdgpu_ps float @load_i8(ptr addrspace(8) inreg %rsrc) {
  %data = call i8 @llvm.amdgcn.struct.ptr.buffer.load.format.i8(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %zext = zext i8 %data to i32
  %fdata = bitcast i32 %zext to float
  ret float %fdata
}

; CHECK: error: {{.*}}unsupported sub-dword format buffer store
define amdgpu_ps void @store_i8(ptr addrspace(8) inreg %rsrc, i8 %data, i32 %index) {
  call void @llvm.amdgcn.struct.ptr.buffer.store.format.i8(i8 %data, ptr addrspace(8) %rsrc, i32 %index, i32 0, i32 0, i32 0)
  ret void
}
