; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=FUNC %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone

; Make sure we don't turn the 32-bit argument load into a 16-bit
; load. There aren't extending scalar lods, so that would require
; using a buffer_load instruction.

; FUNC-LABEL: {{^}}truncate_kernarg_i32_to_i16:
; SI: s_load_dword s
; SI: buffer_store_short v
define amdgpu_kernel void @truncate_kernarg_i32_to_i16(i16 addrspace(1)* %out, i32 %arg) nounwind {
  %trunc = trunc i32 %arg to i16
  store i16 %trunc, i16 addrspace(1)* %out
  ret void
}

; It should be OK (and probably performance neutral) to reduce this,
; but we don't know if the load is uniform yet.

; FUNC-LABEL: {{^}}truncate_buffer_load_i32_to_i16:
; SI: buffer_load_dword v
; SI: buffer_store_short v
define amdgpu_kernel void @truncate_buffer_load_i32_to_i16(i16 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i16, i16 addrspace(1)* %out, i32 %tid
  %load = load i32, i32 addrspace(1)* %gep.in
  %trunc = trunc i32 %load to i16
  store i16 %trunc, i16 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}truncate_kernarg_i32_to_i8:
; SI: s_load_dword s
; SI: buffer_store_byte v
define amdgpu_kernel void @truncate_kernarg_i32_to_i8(i8 addrspace(1)* %out, i32 %arg) nounwind {
  %trunc = trunc i32 %arg to i8
  store i8 %trunc, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}truncate_buffer_load_i32_to_i8:
; SI: buffer_load_dword v
; SI: buffer_store_byte v
define amdgpu_kernel void @truncate_buffer_load_i32_to_i8(i8 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i8, i8 addrspace(1)* %out, i32 %tid
  %load = load i32, i32 addrspace(1)* %gep.in
  %trunc = trunc i32 %load to i8
  store i8 %trunc, i8 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}truncate_kernarg_i32_to_i1:
; SI: s_load_dword s
; SI: buffer_store_byte v
define amdgpu_kernel void @truncate_kernarg_i32_to_i1(i1 addrspace(1)* %out, i32 %arg) nounwind {
  %trunc = trunc i32 %arg to i1
  store i1 %trunc, i1 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}truncate_buffer_load_i32_to_i1:
; SI: buffer_load_dword v
; SI: buffer_store_byte v
define amdgpu_kernel void @truncate_buffer_load_i32_to_i1(i1 addrspace(1)* %out, i32 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.in = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i1, i1 addrspace(1)* %out, i32 %tid
  %load = load i32, i32 addrspace(1)* %gep.in
  %trunc = trunc i32 %load to i1
  store i1 %trunc, i1 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}truncate_kernarg_i64_to_i32:
; SI: s_load_dword s
; SI: buffer_store_dword v
define amdgpu_kernel void @truncate_kernarg_i64_to_i32(i32 addrspace(1)* %out, i64 %arg) nounwind {
  %trunc = trunc i64 %arg to i32
  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}truncate_buffer_load_i64_to_i32:
; SI: buffer_load_dword v
; SI: buffer_store_dword v
define amdgpu_kernel void @truncate_buffer_load_i64_to_i32(i32 addrspace(1)* %out, i64 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %load = load i64, i64 addrspace(1)* %gep.in
  %trunc = trunc i64 %load to i32
  store i32 %trunc, i32 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}srl_kernarg_i64_to_i32:
; SI: s_load_dword s
; SI: buffer_store_dword v
define amdgpu_kernel void @srl_kernarg_i64_to_i32(i32 addrspace(1)* %out, i64 %arg) nounwind {
  %srl = lshr i64 %arg, 32
  %trunc = trunc i64 %srl to i32
  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}srl_buffer_load_i64_to_i32:
; SI: buffer_load_dword v
; SI: buffer_store_dword v
define amdgpu_kernel void @srl_buffer_load_i64_to_i32(i32 addrspace(1)* %out, i64 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i32, i32 addrspace(1)* %out, i32 %tid
  %load = load i64, i64 addrspace(1)* %gep.in
  %srl = lshr i64 %load, 32
  %trunc = trunc i64 %srl to i32
  store i32 %trunc, i32 addrspace(1)* %gep.out
  ret void
}

; Might as well reduce to 8-bit loads.
; FUNC-LABEL: {{^}}truncate_kernarg_i16_to_i8:
; SI: s_load_dword s
; SI: buffer_store_byte v
define amdgpu_kernel void @truncate_kernarg_i16_to_i8(i8 addrspace(1)* %out, i16 %arg) nounwind {
  %trunc = trunc i16 %arg to i8
  store i8 %trunc, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}truncate_buffer_load_i16_to_i8:
; SI: buffer_load_ubyte v
; SI: buffer_store_byte v
define amdgpu_kernel void @truncate_buffer_load_i16_to_i8(i8 addrspace(1)* %out, i16 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.in = getelementptr i16, i16 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i8, i8 addrspace(1)* %out, i32 %tid
  %load = load i16, i16 addrspace(1)* %gep.in
  %trunc = trunc i16 %load to i8
  store i8 %trunc, i8 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}srl_kernarg_i64_to_i8:
; SI: s_load_dword s
; SI: buffer_store_byte v
define amdgpu_kernel void @srl_kernarg_i64_to_i8(i8 addrspace(1)* %out, i64 %arg) nounwind {
  %srl = lshr i64 %arg, 32
  %trunc = trunc i64 %srl to i8
  store i8 %trunc, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}srl_buffer_load_i64_to_i8:
; SI: buffer_load_dword v
; SI: buffer_store_byte v
define amdgpu_kernel void @srl_buffer_load_i64_to_i8(i8 addrspace(1)* %out, i64 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i8, i8 addrspace(1)* %out, i32 %tid
  %load = load i64, i64 addrspace(1)* %gep.in
  %srl = lshr i64 %load, 32
  %trunc = trunc i64 %srl to i8
  store i8 %trunc, i8 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}truncate_kernarg_i64_to_i8:
; SI: s_load_dword s
; SI: buffer_store_byte v
define amdgpu_kernel void @truncate_kernarg_i64_to_i8(i8 addrspace(1)* %out, i64 %arg) nounwind {
  %trunc = trunc i64 %arg to i8
  store i8 %trunc, i8 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}truncate_buffer_load_i64_to_i8:
; SI: buffer_load_dword v
; SI: buffer_store_byte v
define amdgpu_kernel void @truncate_buffer_load_i64_to_i8(i8 addrspace(1)* %out, i64 addrspace(1)* %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
  %gep.in = getelementptr i64, i64 addrspace(1)* %in, i32 %tid
  %gep.out = getelementptr i8, i8 addrspace(1)* %out, i32 %tid
  %load = load i64, i64 addrspace(1)* %gep.in
  %trunc = trunc i64 %load to i8
  store i8 %trunc, i8 addrspace(1)* %gep.out
  ret void
}

; FUNC-LABEL: {{^}}smrd_mask_i32_to_i16
; SI: s_load_dword [[LOAD:s[0-9]+]], s[{{[0-9]+}}:{{[0-9]+}}], 0x0
; SI: s_waitcnt lgkmcnt(0)
; SI: s_and_b32 s{{[0-9]+}}, [[LOAD]], 0xffff
define amdgpu_kernel void @smrd_mask_i32_to_i16(i32 addrspace(1)* %out, i32 addrspace(2)* %in) {
entry:
  %val = load i32, i32 addrspace(2)* %in
  %mask = and i32 %val, 65535
  store i32 %mask, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}extract_hi_i64_bitcast_v2i32:
; SI: buffer_load_dword v
; SI: buffer_store_dword v
define amdgpu_kernel void @extract_hi_i64_bitcast_v2i32(i32 addrspace(1)* %out, <2 x i32> addrspace(1)* %in) nounwind {
  %ld = load <2 x i32>, <2 x i32> addrspace(1)* %in
  %bc = bitcast <2 x i32> %ld to i64
  %hi = lshr i64 %bc, 32
  %trunc = trunc i64 %hi to i32
  store i32 %trunc, i32 addrspace(1)* %out
  ret void
}
