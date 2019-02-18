; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -amdgpu-ir-lower-kernel-arguments=0 -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefixes=VI,GCN,HSA-VI,FUNC %s

; Repeat of some problematic tests in kernel-args.ll, with the IR
; argument lowering pass disabled. Struct padding needs to be
; accounted for, as well as legalization of types changing offsets.

; FUNC-LABEL: {{^}}i1_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4

; GCN: s_load_dword s
; GCN: s_and_b32
define amdgpu_kernel void @i1_arg(i1 addrspace(1)* %out, i1 %x) nounwind {
  store i1 %x, i1 addrspace(1)* %out, align 1
  ret void
}

; FUNC-LABEL: {{^}}v3i8_arg:
; HSA-VI: kernarg_segment_byte_size = 12
; HSA-VI: kernarg_segment_alignment = 4
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x0
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x8
define amdgpu_kernel void @v3i8_arg(<3 x i8> addrspace(1)* nocapture %out, <3 x i8> %in) nounwind {
entry:
  store <3 x i8> %in, <3 x i8> addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}i65_arg:
; HSA-VI: kernarg_segment_byte_size = 24
; HSA-VI: kernarg_segment_alignment = 4
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x0
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
define amdgpu_kernel void @i65_arg(i65 addrspace(1)* nocapture %out, i65 %in) nounwind {
entry:
  store i65 %in, i65 addrspace(1)* %out, align 4
  ret void
}

; FUNC-LABEL: {{^}}empty_struct_arg:
; HSA-VI: kernarg_segment_byte_size = 0
define amdgpu_kernel void @empty_struct_arg({} %in) nounwind {
  ret void
}

; The correct load offsets for these:
; load 4 from 0,
; load 8 from 8
; load 4 from 24
; load 8 from 32

; With the SelectionDAG argument lowering, the alignments for the
; struct members is not properly considered, making these wrong.

; FIXME: Total argument size is computed wrong
; FUNC-LABEL: {{^}}struct_argument_alignment:
; HSA-VI: kernarg_segment_byte_size = 40
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x18
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x20
define amdgpu_kernel void @struct_argument_alignment({i32, i64} %arg0, i8, {i32, i64} %arg1) {
  %val0 = extractvalue {i32, i64} %arg0, 0
  %val1 = extractvalue {i32, i64} %arg0, 1
  %val2 = extractvalue {i32, i64} %arg1, 0
  %val3 = extractvalue {i32, i64} %arg1, 1
  store volatile i32 %val0, i32 addrspace(1)* null
  store volatile i64 %val1, i64 addrspace(1)* null
  store volatile i32 %val2, i32 addrspace(1)* null
  store volatile i64 %val3, i64 addrspace(1)* null
  ret void
}

; No padding between i8 and next struct, but round up at end to 4 byte
; multiple.
; FUNC-LABEL: {{^}}packed_struct_argument_alignment:
; HSA-VI: kernarg_segment_byte_size = 28
; HSA-VI: global_load_dword v{{[0-9]+}}, v{{\[[0-9]+:[0-9]+\]}}, off offset:13
; HSA-VI: global_load_dwordx2 v{{\[[0-9]+:[0-9]+\]}}, v{{\[[0-9]+:[0-9]+\]}}, off offset:17
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x4
define amdgpu_kernel void @packed_struct_argument_alignment(<{i32, i64}> %arg0, i8, <{i32, i64}> %arg1) {
  %val0 = extractvalue <{i32, i64}> %arg0, 0
  %val1 = extractvalue <{i32, i64}> %arg0, 1
  %val2 = extractvalue <{i32, i64}> %arg1, 0
  %val3 = extractvalue <{i32, i64}> %arg1, 1
  store volatile i32 %val0, i32 addrspace(1)* null
  store volatile i64 %val1, i64 addrspace(1)* null
  store volatile i32 %val2, i32 addrspace(1)* null
  store volatile i64 %val3, i64 addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}struct_argument_alignment_after:
; HSA-VI: kernarg_segment_byte_size = 64
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x8
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x18
; HSA-VI: s_load_dwordx2 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x20
; HSA-VI: s_load_dwordx4 s{{\[[0-9]+:[0-9]+\]}}, s[4:5], 0x30
define amdgpu_kernel void @struct_argument_alignment_after({i32, i64} %arg0, i8, {i32, i64} %arg2, i8, <4 x i32> %arg4) {
  %val0 = extractvalue {i32, i64} %arg0, 0
  %val1 = extractvalue {i32, i64} %arg0, 1
  %val2 = extractvalue {i32, i64} %arg2, 0
  %val3 = extractvalue {i32, i64} %arg2, 1
  store volatile i32 %val0, i32 addrspace(1)* null
  store volatile i64 %val1, i64 addrspace(1)* null
  store volatile i32 %val2, i32 addrspace(1)* null
  store volatile i64 %val3, i64 addrspace(1)* null
  store volatile <4 x i32> %arg4, <4 x i32> addrspace(1)* null
  ret void
}

; GCN-LABEL: {{^}}array_3xi32:
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x4
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x8
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0xc
define amdgpu_kernel void @array_3xi32(i16 %arg0, [3 x i32] %arg1) {
  store volatile i16 %arg0, i16 addrspace(1)* undef
  store volatile [3 x i32] %arg1, [3 x i32] addrspace(1)* undef
  ret void
}

; GCN-LABEL: {{^}}array_3xi16:
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x0
; HSA-VI: s_load_dword s{{[0-9]+}}, s[4:5], 0x4
define amdgpu_kernel void @array_3xi16(i8 %arg0, [3 x i16] %arg1) {
  store volatile i8 %arg0, i8 addrspace(1)* undef
  store volatile [3 x i16] %arg1, [3 x i16] addrspace(1)* undef
  ret void
}
