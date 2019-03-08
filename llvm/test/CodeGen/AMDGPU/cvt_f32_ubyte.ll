; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -allow-deprecated-dag-overlap -check-prefix=GCN -check-prefix=VI %s

declare i32 @llvm.amdgcn.workitem.id.x() nounwind readnone
declare i32 @llvm.amdgcn.workitem.id.y() nounwind readnone

; GCN-LABEL: {{^}}load_i8_to_f32:
; GCN: {{buffer|flat}}_load_ubyte [[LOADREG:v[0-9]+]],
; GCN-NOT: bfe
; GCN-NOT: lshr
; GCN: v_cvt_f32_ubyte0_e32 [[CONV:v[0-9]+]], [[LOADREG]]
; GCN: buffer_store_dword [[CONV]],
define amdgpu_kernel void @load_i8_to_f32(float addrspace(1)* noalias %out, i8 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i8, i8 addrspace(1)* %in, i32 %tid
  %load = load i8, i8 addrspace(1)* %gep, align 1
  %cvt = uitofp i8 %load to float
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}load_v2i8_to_v2f32:
; GCN: {{buffer|flat}}_load_ushort [[LD:v[0-9]+]]
; GCN-DAG: v_cvt_f32_ubyte1_e32 v[[HIRESULT:[0-9]+]], [[LD]]
; GCN-DAG: v_cvt_f32_ubyte0_e32 v[[LORESULT:[0-9]+]], [[LD]]
; GCN: buffer_store_dwordx2 v{{\[}}[[LORESULT]]:[[HIRESULT]]{{\]}},
define amdgpu_kernel void @load_v2i8_to_v2f32(<2 x float> addrspace(1)* noalias %out, <2 x i8> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <2 x i8>, <2 x i8> addrspace(1)* %in, i32 %tid
  %load = load <2 x i8>, <2 x i8> addrspace(1)* %gep, align 2
  %cvt = uitofp <2 x i8> %load to <2 x float>
  store <2 x float> %cvt, <2 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}load_v3i8_to_v3f32:
; GCN: {{buffer|flat}}_load_dword [[VAL:v[0-9]+]]
; GCN-NOT: v_cvt_f32_ubyte3_e32
; GCN-DAG: v_cvt_f32_ubyte2_e32 v[[HIRESULT:[0-9]+]], [[VAL]]
; GCN-DAG: v_cvt_f32_ubyte1_e32 v[[MDRESULT:[0-9]+]], [[VAL]]
; GCN-DAG: v_cvt_f32_ubyte0_e32 v[[LORESULT:[0-9]+]], [[VAL]]
; SI: buffer_store_dwordx2 v{{\[}}[[LORESULT]]:[[MDRESULT]]{{\]}},
; VI: buffer_store_dwordx3 v{{\[}}[[LORESULT]]:[[HIRESULT]]{{\]}},
define amdgpu_kernel void @load_v3i8_to_v3f32(<3 x float> addrspace(1)* noalias %out, <3 x i8> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <3 x i8>, <3 x i8> addrspace(1)* %in, i32 %tid
  %load = load <3 x i8>, <3 x i8> addrspace(1)* %gep, align 4
  %cvt = uitofp <3 x i8> %load to <3 x float>
  store <3 x float> %cvt, <3 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}load_v4i8_to_v4f32:
; GCN: {{buffer|flat}}_load_dword [[LOADREG:v[0-9]+]]
; GCN-NOT: bfe
; GCN-NOT: lshr
; GCN-DAG: v_cvt_f32_ubyte3_e32 v[[HIRESULT:[0-9]+]], [[LOADREG]]
; GCN-DAG: v_cvt_f32_ubyte2_e32 v{{[0-9]+}}, [[LOADREG]]
; GCN-DAG: v_cvt_f32_ubyte1_e32 v{{[0-9]+}}, [[LOADREG]]
; GCN-DAG: v_cvt_f32_ubyte0_e32 v[[LORESULT:[0-9]+]], [[LOADREG]]
; GCN: buffer_store_dwordx4 v{{\[}}[[LORESULT]]:[[HIRESULT]]{{\]}},
define amdgpu_kernel void @load_v4i8_to_v4f32(<4 x float> addrspace(1)* noalias %out, <4 x i8> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %gep, align 4
  %cvt = uitofp <4 x i8> %load to <4 x float>
  store <4 x float> %cvt, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; This should not be adding instructions to shift into the correct
; position in the word for the component.

; FIXME: Packing bytes
; GCN-LABEL: {{^}}load_v4i8_to_v4f32_unaligned:
; GCN: {{buffer|flat}}_load_ubyte [[LOADREG3:v[0-9]+]]
; GCN: {{buffer|flat}}_load_ubyte [[LOADREG2:v[0-9]+]]
; GCN: {{buffer|flat}}_load_ubyte [[LOADREG1:v[0-9]+]]
; GCN: {{buffer|flat}}_load_ubyte [[LOADREG0:v[0-9]+]]
; GCN-DAG: v_lshlrev_b32
; GCN-DAG: v_or_b32
; GCN-DAG: v_cvt_f32_ubyte0_e32 v[[LORESULT:[0-9]+]],
; GCN-DAG: v_cvt_f32_ubyte0_e32 v{{[0-9]+}},
; GCN-DAG: v_cvt_f32_ubyte0_e32 v{{[0-9]+}},
; GCN-DAG: v_cvt_f32_ubyte0_e32 v[[HIRESULT:[0-9]+]]

; GCN: buffer_store_dwordx4
define amdgpu_kernel void @load_v4i8_to_v4f32_unaligned(<4 x float> addrspace(1)* noalias %out, <4 x i8> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %gep, align 1
  %cvt = uitofp <4 x i8> %load to <4 x float>
  store <4 x float> %cvt, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; FIXME: Need to handle non-uniform case for function below (load without gep).
; Instructions still emitted to repack bytes for add use.

; GCN-LABEL: {{^}}load_v4i8_to_v4f32_2_uses:
; GCN: {{buffer|flat}}_load_dword
; GCN-DAG: v_cvt_f32_ubyte0_e32
; GCN-DAG: v_cvt_f32_ubyte1_e32
; GCN-DAG: v_cvt_f32_ubyte2_e32
; GCN-DAG: v_cvt_f32_ubyte3_e32

; GCN-DAG: v_lshrrev_b32_e32 v{{[0-9]+}}, 24

; SI-DAG: v_lshlrev_b32_e32 v{{[0-9]+}}, 16
; SI-DAG: v_lshlrev_b32_e32 v{{[0-9]+}}, 8
; SI-DAG: v_and_b32_e32 v{{[0-9]+}}, 0xffff,
; SI-DAG: v_and_b32_e32 v{{[0-9]+}}, 0xff00,
; SI-DAG: v_add_i32

; VI-DAG: v_and_b32_e32 v{{[0-9]+}}, 0xffffff00,
; VI-DAG: v_add_u16_e32
; VI-DAG: v_add_u16_e32

; GCN-DAG: {{buffer|flat}}_store_dwordx4
; GCN-DAG: {{buffer|flat}}_store_dword

; GCN: s_endpgm
define amdgpu_kernel void @load_v4i8_to_v4f32_2_uses(<4 x float> addrspace(1)* noalias %out, <4 x i8> addrspace(1)* noalias %out2, <4 x i8> addrspace(1)* noalias %in) nounwind {
  %tid.x = call i32 @llvm.amdgcn.workitem.id.x()
  %in.ptr = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid.x
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %in.ptr, align 4
  %cvt = uitofp <4 x i8> %load to <4 x float>
  store <4 x float> %cvt, <4 x float> addrspace(1)* %out, align 16
  %add = add <4 x i8> %load, <i8 9, i8 9, i8 9, i8 9> ; Second use of %load
  store <4 x i8> %add, <4 x i8> addrspace(1)* %out2, align 4
  ret void
}

; Make sure this doesn't crash.
; GCN-LABEL: {{^}}load_v7i8_to_v7f32:
; GCN: s_endpgm
define amdgpu_kernel void @load_v7i8_to_v7f32(<7 x float> addrspace(1)* noalias %out, <7 x i8> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <7 x i8>, <7 x i8> addrspace(1)* %in, i32 %tid
  %load = load <7 x i8>, <7 x i8> addrspace(1)* %gep, align 1
  %cvt = uitofp <7 x i8> %load to <7 x float>
  store <7 x float> %cvt, <7 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}load_v8i8_to_v8f32:
; GCN: {{buffer|flat}}_load_dwordx2 v{{\[}}[[LOLOAD:[0-9]+]]:[[HILOAD:[0-9]+]]{{\]}},
; GCN-NOT: bfe
; GCN-NOT: lshr
; GCN-DAG: v_cvt_f32_ubyte3_e32 v{{[0-9]+}}, v[[LOLOAD]]
; GCN-DAG: v_cvt_f32_ubyte2_e32 v{{[0-9]+}}, v[[LOLOAD]]
; GCN-DAG: v_cvt_f32_ubyte1_e32 v{{[0-9]+}}, v[[LOLOAD]]
; GCN-DAG: v_cvt_f32_ubyte0_e32 v{{[0-9]+}}, v[[LOLOAD]]
; GCN-DAG: v_cvt_f32_ubyte3_e32 v{{[0-9]+}}, v[[HILOAD]]
; GCN-DAG: v_cvt_f32_ubyte2_e32 v{{[0-9]+}}, v[[HILOAD]]
; GCN-DAG: v_cvt_f32_ubyte1_e32 v{{[0-9]+}}, v[[HILOAD]]
; GCN-DAG: v_cvt_f32_ubyte0_e32 v{{[0-9]+}}, v[[HILOAD]]
; GCN-NOT: bfe
; GCN-NOT: lshr
; GCN: buffer_store_dwordx4
; GCN: buffer_store_dwordx4
define amdgpu_kernel void @load_v8i8_to_v8f32(<8 x float> addrspace(1)* noalias %out, <8 x i8> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <8 x i8>, <8 x i8> addrspace(1)* %in, i32 %tid
  %load = load <8 x i8>, <8 x i8> addrspace(1)* %gep, align 8
  %cvt = uitofp <8 x i8> %load to <8 x float>
  store <8 x float> %cvt, <8 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}i8_zext_inreg_i32_to_f32:
; GCN: {{buffer|flat}}_load_dword [[LOADREG:v[0-9]+]],
; GCN: v_add_{{[iu]}}32_e32 [[ADD:v[0-9]+]], vcc, 2, [[LOADREG]]
; GCN-NEXT: v_cvt_f32_ubyte0_e32 [[CONV:v[0-9]+]], [[ADD]]
; GCN: buffer_store_dword [[CONV]],
define amdgpu_kernel void @i8_zext_inreg_i32_to_f32(float addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %load = load i32, i32 addrspace(1)* %gep, align 4
  %add = add i32 %load, 2
  %inreg = and i32 %add, 255
  %cvt = uitofp i32 %inreg to float
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}i8_zext_inreg_hi1_to_f32:
define amdgpu_kernel void @i8_zext_inreg_hi1_to_f32(float addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %load = load i32, i32 addrspace(1)* %gep, align 4
  %inreg = and i32 %load, 65280
  %shr = lshr i32 %inreg, 8
  %cvt = uitofp i32 %shr to float
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; We don't get these ones because of the zext, but instcombine removes
; them so it shouldn't really matter.
; GCN-LABEL: {{^}}i8_zext_i32_to_f32:
define amdgpu_kernel void @i8_zext_i32_to_f32(float addrspace(1)* noalias %out, i8 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i8, i8 addrspace(1)* %in, i32 %tid
  %load = load i8, i8 addrspace(1)* %gep, align 1
  %ext = zext i8 %load to i32
  %cvt = uitofp i32 %ext to float
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; GCN-LABEL: {{^}}v4i8_zext_v4i32_to_v4f32:
define amdgpu_kernel void @v4i8_zext_v4i32_to_v4f32(<4 x float> addrspace(1)* noalias %out, <4 x i8> addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr <4 x i8>, <4 x i8> addrspace(1)* %in, i32 %tid
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %gep, align 1
  %ext = zext <4 x i8> %load to <4 x i32>
  %cvt = uitofp <4 x i32> %ext to <4 x float>
  store <4 x float> %cvt, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; GCN-LABEL: {{^}}extract_byte0_to_f32:
; GCN: {{buffer|flat}}_load_dword [[VAL:v[0-9]+]]
; GCN-NOT: [[VAL]]
; GCN: v_cvt_f32_ubyte0_e32 [[CONV:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[CONV]]
define amdgpu_kernel void @extract_byte0_to_f32(float addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep
  %and = and i32 %val, 255
  %cvt = uitofp i32 %and to float
  store float %cvt, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}extract_byte1_to_f32:
; GCN: {{buffer|flat}}_load_dword [[VAL:v[0-9]+]]
; GCN-NOT: [[VAL]]
; GCN: v_cvt_f32_ubyte1_e32 [[CONV:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[CONV]]
define amdgpu_kernel void @extract_byte1_to_f32(float addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep
  %srl = lshr i32 %val, 8
  %and = and i32 %srl, 255
  %cvt = uitofp i32 %and to float
  store float %cvt, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}extract_byte2_to_f32:
; GCN: {{buffer|flat}}_load_dword [[VAL:v[0-9]+]]
; GCN-NOT: [[VAL]]
; GCN: v_cvt_f32_ubyte2_e32 [[CONV:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[CONV]]
define amdgpu_kernel void @extract_byte2_to_f32(float addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep
  %srl = lshr i32 %val, 16
  %and = and i32 %srl, 255
  %cvt = uitofp i32 %and to float
  store float %cvt, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}extract_byte3_to_f32:
; GCN: {{buffer|flat}}_load_dword [[VAL:v[0-9]+]]
; GCN-NOT: [[VAL]]
; GCN: v_cvt_f32_ubyte3_e32 [[CONV:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[CONV]]
define amdgpu_kernel void @extract_byte3_to_f32(float addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %tid = call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr i32, i32 addrspace(1)* %in, i32 %tid
  %val = load i32, i32 addrspace(1)* %gep
  %srl = lshr i32 %val, 24
  %and = and i32 %srl, 255
  %cvt = uitofp i32 %and to float
  store float %cvt, float addrspace(1)* %out
  ret void
}

; GCN-LABEL: {{^}}cvt_ubyte0_or_multiuse:
; GCN:     {{buffer|flat}}_load_dword [[LOADREG:v[0-9]+]],
; GCN-DAG: v_or_b32_e32 [[OR:v[0-9]+]], 0x80000001, [[LOADREG]]
; GCN-DAG: v_cvt_f32_ubyte0_e32 [[CONV:v[0-9]+]], [[OR]]
; GCN:     v_add_f32_e32 [[RES:v[0-9]+]], [[OR]], [[CONV]]
; GCN:     buffer_store_dword [[RES]],
define amdgpu_kernel void @cvt_ubyte0_or_multiuse(i32 addrspace(1)* %in, float addrspace(1)* %out) {
bb:
  %lid = tail call i32 @llvm.amdgcn.workitem.id.x()
  %gep = getelementptr inbounds i32, i32 addrspace(1)* %in, i32 %lid
  %load = load i32, i32 addrspace(1)* %gep
  %or = or i32 %load, -2147483647
  %and = and i32 %or, 255
  %uitofp = uitofp i32 %and to float
  %cast = bitcast i32 %or to float
  %add = fadd float %cast, %uitofp
  store float %add, float addrspace(1)* %out
  ret void
}
