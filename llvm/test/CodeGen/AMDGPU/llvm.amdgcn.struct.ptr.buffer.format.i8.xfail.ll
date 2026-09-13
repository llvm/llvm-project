; RUN: not llc -global-isel=0 -mtriple=amdgpu9.00 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel -mtriple=amdgpu9.00 -filetype=null %s 2>&1 | FileCheck %s
; RUN: not llc -global-isel=0 -mtriple=amdgpu9.0a -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck -check-prefix=GFX90A %s
; RUN: not llc -global-isel -mtriple=amdgpu9.0a -mcpu=gfx90a -filetype=null %s 2>&1 | FileCheck -check-prefix=GFX90A %s

; An i8 buffer.load.format / buffer.store.format has no corresponding real
; instruction (no byte-granularity format access exists in hardware), so both
; SelectionDAG and GlobalISel must refuse to lower it.

; CHECK: error: {{.*}}unsupported sub-dword format buffer load
; GFX90A: error: {{.*}}unsupported sub-dword format buffer load
define amdgpu_ps float @load_i8(ptr addrspace(8) inreg %rsrc) {
  %data = call i8 @llvm.amdgcn.struct.ptr.buffer.load.format.i8(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %zext = zext i8 %data to i32
  %fdata = bitcast i32 %zext to float
  ret float %fdata
}

; CHECK: error: {{.*}}unsupported sub-dword format buffer store
; GFX90A: error: {{.*}}unsupported sub-dword format buffer store
define amdgpu_ps void @store_i8(ptr addrspace(8) inreg %rsrc, i8 %data, i32 %index) {
  call void @llvm.amdgcn.struct.ptr.buffer.store.format.i8(i8 %data, ptr addrspace(8) %rsrc, i32 %index, i32 0, i32 0, i32 0)
  ret void
}

; D16 buffer.load.format combined with TFE has no real hardware encoding on
; gfx90a, so it must be refused there. Other targets (gfx8/gfx10/gfx11/gfx12)
; have real TFE encodings and are covered by
; llvm.amdgcn.struct.ptr.buffer.load.format.d16.tfe.ll instead.
; CHECK-NOT: error: {{.*}}TFE D16 format buffer load
; GFX90A: error: {{.*}}TFE D16 format buffer load is not supported on this GPU
define amdgpu_kernel void @load_v3i16_tfe(ptr addrspace(8) inreg %rsrc, ptr addrspace(1) %out, ptr addrspace(1) %status) {
  %r = call {<3 x i16>, i32} @llvm.amdgcn.struct.ptr.buffer.load.format.sl_v3i16i32s(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %data = extractvalue {<3 x i16>, i32} %r, 0
  %st = extractvalue {<3 x i16>, i32} %r, 1
  store <3 x i16> %data, ptr addrspace(1) %out
  store i32 %st, ptr addrspace(1) %status
  ret void
}

; GFX90A: error: {{.*}}TFE D16 format buffer load is not supported on this GPU
define amdgpu_kernel void @load_f16_tfe(ptr addrspace(8) inreg %rsrc, ptr addrspace(1) %out, ptr addrspace(1) %status) {
  %r = call {half, i32} @llvm.amdgcn.struct.ptr.buffer.load.format.sl_f16i32s(ptr addrspace(8) %rsrc, i32 0, i32 0, i32 0, i32 0)
  %data = extractvalue {half, i32} %r, 0
  %st = extractvalue {half, i32} %r, 1
  store half %data, ptr addrspace(1) %out
  store i32 %st, ptr addrspace(1) %status
  ret void
}
