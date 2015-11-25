; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}load_i8_to_f32:
; SI: buffer_load_ubyte [[LOADREG:v[0-9]+]],
; SI-NOT: bfe
; SI-NOT: lshr
; SI: v_cvt_f32_ubyte0_e32 [[CONV:v[0-9]+]], [[LOADREG]]
; SI: buffer_store_dword [[CONV]],
define void @load_i8_to_f32(float addrspace(1)* noalias %out, i8 addrspace(1)* noalias %in) nounwind {
  %load = load i8, i8 addrspace(1)* %in, align 1
  %cvt = uitofp i8 %load to float
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}load_v2i8_to_v2f32:
; SI: buffer_load_ushort [[LOADREG:v[0-9]+]],
; SI-NOT: bfe
; SI-NOT: lshr
; SI-NOT: and
; SI-DAG: v_cvt_f32_ubyte1_e32 v[[HIRESULT:[0-9]+]], [[LOADREG]]
; SI-DAG: v_cvt_f32_ubyte0_e32 v[[LORESULT:[0-9]+]], [[LOADREG]]
; SI: buffer_store_dwordx2 v{{\[}}[[LORESULT]]:[[HIRESULT]]{{\]}},
define void @load_v2i8_to_v2f32(<2 x float> addrspace(1)* noalias %out, <2 x i8> addrspace(1)* noalias %in) nounwind {
  %load = load <2 x i8>, <2 x i8> addrspace(1)* %in, align 2
  %cvt = uitofp <2 x i8> %load to <2 x float>
  store <2 x float> %cvt, <2 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}load_v3i8_to_v3f32:
; SI-NOT: bfe
; SI-NOT: v_cvt_f32_ubyte3_e32
; SI-DAG: v_cvt_f32_ubyte2_e32
; SI-DAG: v_cvt_f32_ubyte1_e32
; SI-DAG: v_cvt_f32_ubyte0_e32
; SI: buffer_store_dwordx2 v{{\[}}[[LORESULT]]:[[HIRESULT]]{{\]}},
define void @load_v3i8_to_v3f32(<3 x float> addrspace(1)* noalias %out, <3 x i8> addrspace(1)* noalias %in) nounwind {
  %load = load <3 x i8>, <3 x i8> addrspace(1)* %in, align 4
  %cvt = uitofp <3 x i8> %load to <3 x float>
  store <3 x float> %cvt, <3 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}load_v4i8_to_v4f32:
; SI: buffer_load_dword [[LOADREG:v[0-9]+]]
; SI-NOT: bfe
; SI-NOT: lshr
; SI-DAG: v_cvt_f32_ubyte3_e32 v[[HIRESULT:[0-9]+]], [[LOADREG]]
; SI-DAG: v_cvt_f32_ubyte2_e32 v{{[0-9]+}}, [[LOADREG]]
; SI-DAG: v_cvt_f32_ubyte1_e32 v{{[0-9]+}}, [[LOADREG]]
; SI-DAG: v_cvt_f32_ubyte0_e32 v[[LORESULT:[0-9]+]], [[LOADREG]]
; SI: buffer_store_dwordx4 v{{\[}}[[LORESULT]]:[[HIRESULT]]{{\]}},
define void @load_v4i8_to_v4f32(<4 x float> addrspace(1)* noalias %out, <4 x i8> addrspace(1)* noalias %in) nounwind {
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  %cvt = uitofp <4 x i8> %load to <4 x float>
  store <4 x float> %cvt, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; This should not be adding instructions to shift into the correct
; position in the word for the component.

; SI-LABEL: {{^}}load_v4i8_to_v4f32_unaligned:
; SI: buffer_load_ubyte [[LOADREG3:v[0-9]+]]
; SI: buffer_load_ubyte [[LOADREG2:v[0-9]+]]
; SI: buffer_load_ubyte [[LOADREG1:v[0-9]+]]
; SI: buffer_load_ubyte [[LOADREG0:v[0-9]+]]
; SI-NOT: v_lshlrev_b32
; SI-NOT: v_or_b32

; SI-DAG: v_cvt_f32_ubyte0_e32 v[[LORESULT:[0-9]+]], [[LOADREG0]]
; SI-DAG: v_cvt_f32_ubyte0_e32 v{{[0-9]+}}, [[LOADREG1]]
; SI-DAG: v_cvt_f32_ubyte0_e32 v{{[0-9]+}}, [[LOADREG2]]
; SI-DAG: v_cvt_f32_ubyte0_e32 v[[HIRESULT:[0-9]+]], [[LOADREG3]]

; SI: buffer_store_dwordx4 v{{\[}}[[LORESULT]]:[[HIRESULT]]{{\]}},
define void @load_v4i8_to_v4f32_unaligned(<4 x float> addrspace(1)* noalias %out, <4 x i8> addrspace(1)* noalias %in) nounwind {
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 1
  %cvt = uitofp <4 x i8> %load to <4 x float>
  store <4 x float> %cvt, <4 x float> addrspace(1)* %out, align 16
  ret void
}

; XXX - This should really still be able to use the v_cvt_f32_ubyte0
; for each component, but computeKnownBits doesn't handle vectors very
; well.

; SI-LABEL: {{^}}load_v4i8_to_v4f32_2_uses:
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: buffer_load_ubyte
; SI: v_cvt_f32_ubyte0_e32
; SI: v_cvt_f32_ubyte0_e32
; SI: v_cvt_f32_ubyte0_e32
; SI: v_cvt_f32_ubyte0_e32

; XXX - replace with this when v4i8 loads aren't scalarized anymore.
; XSI: buffer_load_dword
; XSI: v_cvt_f32_u32_e32
; XSI: v_cvt_f32_u32_e32
; XSI: v_cvt_f32_u32_e32
; XSI: v_cvt_f32_u32_e32
; SI: s_endpgm
define void @load_v4i8_to_v4f32_2_uses(<4 x float> addrspace(1)* noalias %out, <4 x i8> addrspace(1)* noalias %out2, <4 x i8> addrspace(1)* noalias %in) nounwind {
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 4
  %cvt = uitofp <4 x i8> %load to <4 x float>
  store <4 x float> %cvt, <4 x float> addrspace(1)* %out, align 16
  %add = add <4 x i8> %load, <i8 9, i8 9, i8 9, i8 9> ; Second use of %load
  store <4 x i8> %add, <4 x i8> addrspace(1)* %out2, align 4
  ret void
}

; Make sure this doesn't crash.
; SI-LABEL: {{^}}load_v7i8_to_v7f32:
; SI: s_endpgm
define void @load_v7i8_to_v7f32(<7 x float> addrspace(1)* noalias %out, <7 x i8> addrspace(1)* noalias %in) nounwind {
  %load = load <7 x i8>, <7 x i8> addrspace(1)* %in, align 1
  %cvt = uitofp <7 x i8> %load to <7 x float>
  store <7 x float> %cvt, <7 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}load_v8i8_to_v8f32:
; SI: buffer_load_dwordx2 v{{\[}}[[LOLOAD:[0-9]+]]:[[HILOAD:[0-9]+]]{{\]}},
; SI-NOT: bfe
; SI-NOT: lshr
; SI-DAG: v_cvt_f32_ubyte3_e32 v{{[0-9]+}}, v[[LOLOAD]]
; SI-DAG: v_cvt_f32_ubyte2_e32 v{{[0-9]+}}, v[[LOLOAD]]
; SI-DAG: v_cvt_f32_ubyte1_e32 v{{[0-9]+}}, v[[LOLOAD]]
; SI-DAG: v_cvt_f32_ubyte0_e32 v{{[0-9]+}}, v[[LOLOAD]]
; SI-DAG: v_cvt_f32_ubyte3_e32 v{{[0-9]+}}, v[[HILOAD]]
; SI-DAG: v_cvt_f32_ubyte2_e32 v{{[0-9]+}}, v[[HILOAD]]
; SI-DAG: v_cvt_f32_ubyte1_e32 v{{[0-9]+}}, v[[HILOAD]]
; SI-DAG: v_cvt_f32_ubyte0_e32 v{{[0-9]+}}, v[[HILOAD]]
; SI-NOT: bfe
; SI-NOT: lshr
; SI: buffer_store_dwordx4
; SI: buffer_store_dwordx4
define void @load_v8i8_to_v8f32(<8 x float> addrspace(1)* noalias %out, <8 x i8> addrspace(1)* noalias %in) nounwind {
  %load = load <8 x i8>, <8 x i8> addrspace(1)* %in, align 8
  %cvt = uitofp <8 x i8> %load to <8 x float>
  store <8 x float> %cvt, <8 x float> addrspace(1)* %out, align 16
  ret void
}

; SI-LABEL: {{^}}i8_zext_inreg_i32_to_f32:
; SI: buffer_load_dword [[LOADREG:v[0-9]+]],
; SI: v_add_i32_e32 [[ADD:v[0-9]+]], vcc, 2, [[LOADREG]]
; SI-NEXT: v_cvt_f32_ubyte0_e32 [[CONV:v[0-9]+]], [[ADD]]
; SI: buffer_store_dword [[CONV]],
define void @i8_zext_inreg_i32_to_f32(float addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %add = add i32 %load, 2
  %inreg = and i32 %add, 255
  %cvt = uitofp i32 %inreg to float
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

; SI-LABEL: {{^}}i8_zext_inreg_hi1_to_f32:
define void @i8_zext_inreg_hi1_to_f32(float addrspace(1)* noalias %out, i32 addrspace(1)* noalias %in) nounwind {
  %load = load i32, i32 addrspace(1)* %in, align 4
  %inreg = and i32 %load, 65280
  %shr = lshr i32 %inreg, 8
  %cvt = uitofp i32 %shr to float
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}


; We don't get these ones because of the zext, but instcombine removes
; them so it shouldn't really matter.
define void @i8_zext_i32_to_f32(float addrspace(1)* noalias %out, i8 addrspace(1)* noalias %in) nounwind {
  %load = load i8, i8 addrspace(1)* %in, align 1
  %ext = zext i8 %load to i32
  %cvt = uitofp i32 %ext to float
  store float %cvt, float addrspace(1)* %out, align 4
  ret void
}

define void @v4i8_zext_v4i32_to_v4f32(<4 x float> addrspace(1)* noalias %out, <4 x i8> addrspace(1)* noalias %in) nounwind {
  %load = load <4 x i8>, <4 x i8> addrspace(1)* %in, align 1
  %ext = zext <4 x i8> %load to <4 x i32>
  %cvt = uitofp <4 x i32> %ext to <4 x float>
  store <4 x float> %cvt, <4 x float> addrspace(1)* %out, align 16
  ret void
}
