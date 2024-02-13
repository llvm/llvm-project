; RUN:  llc -amdgpu-scalarize-global-loads=false  -mtriple=amdgcn -show-mc-encoding -verify-machineinstrs < %s | FileCheck %s

;;;==========================================================================;;;
;;; MUBUF LOAD TESTS
;;;==========================================================================;;;

; MUBUF load with an immediate byte offset that fits into 12-bits
; CHECK-LABEL: {{^}}mubuf_load0:
; CHECK: buffer_load_dword v{{[0-9]}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4 ; encoding: [0x04,0x00,0x30,0xe0
define amdgpu_kernel void @mubuf_load0(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %0 = getelementptr i32, ptr addrspace(1) %in, i64 1
  %1 = load i32, ptr addrspace(1) %0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; MUBUF load with the largest possible immediate offset
; CHECK-LABEL: {{^}}mubuf_load1:
; CHECK: buffer_load_ubyte v{{[0-9]}}, off, s[{{[0-9]+:[0-9]+}}], 0 offset:4095 ; encoding: [0xff,0x0f,0x20,0xe0
define amdgpu_kernel void @mubuf_load1(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %0 = getelementptr i8, ptr addrspace(1) %in, i64 4095
  %1 = load i8, ptr addrspace(1) %0
  store i8 %1, ptr addrspace(1) %out
  ret void
}

; MUBUF load with an immediate byte offset that doesn't fit into 12-bits
; CHECK-LABEL: {{^}}mubuf_load2:
; CHECK: s_movk_i32 [[SOFFSET:s[0-9]+]], 0x1000
; CHECK: buffer_load_dword v{{[0-9]}}, off, s[{{[0-9]+:[0-9]+}}], [[SOFFSET]] ; encoding: [0x00,0x00,0x30,0xe0
define amdgpu_kernel void @mubuf_load2(ptr addrspace(1) %out, ptr addrspace(1) %in) {
entry:
  %0 = getelementptr i32, ptr addrspace(1) %in, i64 1024
  %1 = load i32, ptr addrspace(1) %0
  store i32 %1, ptr addrspace(1) %out
  ret void
}

; MUBUF load with a 12-bit immediate offset and a register offset
; CHECK-LABEL: {{^}}mubuf_load3:
; CHECK-NOT: ADD
; CHECK: buffer_load_dword v{{[0-9]}}, v[{{[0-9]+:[0-9]+}}], s[{{[0-9]+:[0-9]+}}], 0 addr64 offset:4 ; encoding: [0x04,0x80,0x30,0xe0
define amdgpu_kernel void @mubuf_load3(ptr addrspace(1) %out, ptr addrspace(1) %in, i64 %offset) {
entry:
  %0 = getelementptr i32, ptr addrspace(1) %in, i64 %offset
  %1 = getelementptr i32, ptr addrspace(1) %0, i64 1
  %2 = load i32, ptr addrspace(1) %1
  store i32 %2, ptr addrspace(1) %out
  ret void
}

; CHECK-LABEL: {{^}}soffset_max_imm:
; CHECK: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], 64 offen glc
define amdgpu_gs void @soffset_max_imm(ptr addrspace(4) inreg, ptr addrspace(4) inreg, ptr addrspace(4) inreg, ptr addrspace(4) inreg, i32 inreg, i32 inreg, i32, i32, i32, i32, i32, i32, i32, i32) {
main_body:
  %tmp1 = load ptr addrspace(8), ptr addrspace(4) %0
  %tmp2 = shl i32 %6, 2
  %tmp3 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %tmp1, i32 %tmp2, i32 64, i32 1)
  %tmp4 = add i32 %6, 16
  %tmp1.4xi32 = bitcast ptr addrspace(8) %tmp1 to ptr addrspace(8)
  call void @llvm.amdgcn.raw.ptr.tbuffer.store.i32(i32 %tmp3, ptr addrspace(8) %tmp1.4xi32, i32 %tmp4, i32 %4, i32 68, i32 3)
  ret void
}

; Make sure immediates that aren't inline constants don't get folded into
; the soffset operand.
; FIXME: for this test we should be smart enough to shift the immediate into
; the offset field.
; CHECK-LABEL: {{^}}soffset_no_fold:
; CHECK: s_movk_i32 [[SOFFSET:s[0-9]+]], 0x41
; CHECK: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s[{{[0-9]+}}:{{[0-9]+}}], [[SOFFSET]] offen glc
define amdgpu_gs void @soffset_no_fold(ptr addrspace(4) inreg, ptr addrspace(4) inreg, ptr addrspace(4) inreg, ptr addrspace(4) inreg, i32 inreg, i32 inreg, i32, i32, i32, i32, i32, i32, i32, i32) {
main_body:
  %tmp1 = load ptr addrspace(8), ptr addrspace(4) %0
  %tmp2 = shl i32 %6, 2
  %tmp3 = call i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8) %tmp1, i32 %tmp2, i32 65, i32 1)
  %tmp4 = add i32 %6, 16
  %tmp1.4xi32 = bitcast ptr addrspace(8) %tmp1 to ptr addrspace(8)
  call void @llvm.amdgcn.raw.ptr.tbuffer.store.i32(i32 %tmp3, ptr addrspace(8) %tmp1.4xi32, i32 %tmp4, i32 %4, i32 68, i32 3)
  ret void
}

;;;==========================================================================;;;
;;; MUBUF STORE TESTS
;;;==========================================================================;;;

; MUBUF store with an immediate byte offset that fits into 12-bits
; CHECK-LABEL: {{^}}mubuf_store0:
; CHECK: buffer_store_dword v{{[0-9]}}, off, s[{{[0-9]:[0-9]}}], 0 offset:4 ; encoding: [0x04,0x00,0x70,0xe0
define amdgpu_kernel void @mubuf_store0(ptr addrspace(1) %out) {
entry:
  %0 = getelementptr i32, ptr addrspace(1) %out, i64 1
  store i32 0, ptr addrspace(1) %0
  ret void
}

; MUBUF store with the largest possible immediate offset
; CHECK-LABEL: {{^}}mubuf_store1:
; CHECK: buffer_store_byte v{{[0-9]}}, off, s[{{[0-9]:[0-9]}}], 0 offset:4095 ; encoding: [0xff,0x0f,0x60,0xe0

define amdgpu_kernel void @mubuf_store1(ptr addrspace(1) %out) {
entry:
  %0 = getelementptr i8, ptr addrspace(1) %out, i64 4095
  store i8 0, ptr addrspace(1) %0
  ret void
}

; MUBUF store with an immediate byte offset that doesn't fit into 12-bits
; CHECK-LABEL: {{^}}mubuf_store2:
; CHECK: s_movk_i32 [[SOFFSET:s[0-9]+]], 0x1000
; CHECK: buffer_store_dword v{{[0-9]}}, off, s[{{[0-9]:[0-9]}}], [[SOFFSET]] ; encoding: [0x00,0x00,0x70,0xe0
define amdgpu_kernel void @mubuf_store2(ptr addrspace(1) %out) {
entry:
  %0 = getelementptr i32, ptr addrspace(1) %out, i64 1024
  store i32 0, ptr addrspace(1) %0
  ret void
}

; MUBUF store with a 12-bit immediate offset and a register offset
; CHECK-LABEL: {{^}}mubuf_store3:
; CHECK-NOT: ADD
; CHECK: buffer_store_dword v{{[0-9]}}, v[{{[0-9]:[0-9]}}], s[{{[0-9]:[0-9]}}], 0 addr64 offset:4 ; encoding: [0x04,0x80,0x70,0xe0
define amdgpu_kernel void @mubuf_store3(ptr addrspace(1) %out, i64 %offset) {
entry:
  %0 = getelementptr i32, ptr addrspace(1) %out, i64 %offset
  %1 = getelementptr i32, ptr addrspace(1) %0, i64 1
  store i32 0, ptr addrspace(1) %1
  ret void
}

; CHECK-LABEL: {{^}}store_sgpr_ptr:
; CHECK: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0
define amdgpu_kernel void @store_sgpr_ptr(ptr addrspace(1) %out) {
  store i32 99, ptr addrspace(1) %out, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_sgpr_ptr_offset:
; CHECK: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, 0 offset:40
define amdgpu_kernel void @store_sgpr_ptr_offset(ptr addrspace(1) %out) {
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i32 10
  store i32 99, ptr addrspace(1) %out.gep, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_sgpr_ptr_large_offset:
; CHECK: s_mov_b32 [[SOFFSET:s[0-9]+]], 0x20000
; CHECK: buffer_store_dword v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, [[SOFFSET]]
define amdgpu_kernel void @store_sgpr_ptr_large_offset(ptr addrspace(1) %out) {
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i32 32768
  store i32 99, ptr addrspace(1) %out.gep, align 4
  ret void
}

; CHECK-LABEL: {{^}}store_sgpr_ptr_large_offset_atomic:
; CHECK: s_mov_b32 [[SOFFSET:s[0-9]+]], 0x20000
; CHECK: buffer_atomic_add v{{[0-9]+}}, off, s{{\[[0-9]+:[0-9]+\]}}, [[SOFFSET]]
define amdgpu_kernel void @store_sgpr_ptr_large_offset_atomic(ptr addrspace(1) %out) {
  %gep = getelementptr i32, ptr addrspace(1) %out, i32 32768
  %val = atomicrmw volatile add ptr addrspace(1) %gep, i32 5 syncscope("agent") seq_cst
  ret void
}

; CHECK-LABEL: {{^}}store_vgpr_ptr:
; CHECK: buffer_store_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, s{{\[[0-9]+:[0-9]+\]}}, 0 addr64
define amdgpu_kernel void @store_vgpr_ptr(ptr addrspace(1) %out) {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() readnone
  %out.gep = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 99, ptr addrspace(1) %out.gep, align 4
  ret void
}


declare i32 @llvm.amdgcn.workitem.id.x() #1
declare void @llvm.amdgcn.raw.ptr.tbuffer.store.i32(i32, ptr addrspace(8), i32, i32, i32 immarg, i32 immarg) #2
declare i32 @llvm.amdgcn.raw.ptr.buffer.load.i32(ptr addrspace(8), i32, i32, i32 immarg) #3

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind willreturn writeonly }
attributes #3 = { nounwind readonly willreturn }
attributes #4 = { readnone }
