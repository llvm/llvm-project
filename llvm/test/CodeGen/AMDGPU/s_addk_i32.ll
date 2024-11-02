; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -mtriple=amdgcn--amdpal -mcpu=tonga -mattr=-flat-for-global,-xnack -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; TODO: Some of those tests fail with OS == amdhsa due to unreasonable register
;       allocation differences.

; SI-LABEL: {{^}}s_addk_i32_k0:
; SI: s_load_dword [[VAL:s[0-9]+]]
; SI: s_addk_i32 [[VAL]], 0x41
; SI: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[VRESULT]]
; SI: s_endpgm
define amdgpu_kernel void @s_addk_i32_k0(i32 addrspace(1)* %out, i32 %b) {
  %add = add i32 %b, 65
  store i32 %add, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_addk_i32_k0_x2:
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x41
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x41
; SI: s_endpgm
define amdgpu_kernel void @s_addk_i32_k0_x2(i32 addrspace(1)* %out0, i32 addrspace(1)* %out1, i32 %a, i32 %b) {
  %add0 = add i32 %a, 65
  %add1 = add i32 %b, 65
  store i32 %add0, i32 addrspace(1)* %out0
  store i32 %add1, i32 addrspace(1)* %out1
  ret void
}

; SI-LABEL: {{^}}s_addk_i32_k1:
; SI: s_addk_i32 {{s[0-9]+}}, 0x7fff{{$}}
; SI: s_endpgm
define amdgpu_kernel void @s_addk_i32_k1(i32 addrspace(1)* %out, i32 %b) {
  %add = add i32 %b, 32767 ; (1 << 15) - 1
  store i32 %add, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_addk_i32_k2:
; SI: s_sub_i32 s{{[0-9]+}}, s{{[0-9]+}}, 17
; SI: s_endpgm
define amdgpu_kernel void @s_addk_i32_k2(i32 addrspace(1)* %out, i32 %b) {
  %add = add i32 %b, -17
  store i32 %add, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_addk_i32_k3:
; SI: s_addk_i32 {{s[0-9]+}}, 0xffbf{{$}}
; SI: s_endpgm
define amdgpu_kernel void @s_addk_i32_k3(i32 addrspace(1)* %out, i32 %b) {
  %add = add i32 %b, -65
  store i32 %add, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_addk_v2i32_k0:
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x41
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x42
; SI: s_endpgm
define amdgpu_kernel void @s_addk_v2i32_k0(<2 x i32> addrspace(1)* %out, <2 x i32> %b) {
  %add = add <2 x i32> %b, <i32 65, i32 66>
  store <2 x i32> %add, <2 x i32> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_addk_v4i32_k0:
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x41
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x42
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x43
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x44
; SI: s_endpgm
define amdgpu_kernel void @s_addk_v4i32_k0(<4 x i32> addrspace(1)* %out, <4 x i32> %b) {
  %add = add <4 x i32> %b, <i32 65, i32 66, i32 67, i32 68>
  store <4 x i32> %add, <4 x i32> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_addk_v8i32_k0:
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x41
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x42
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x43
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x44
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x45
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x46
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x47
; SI-DAG: s_addk_i32 {{s[0-9]+}}, 0x48
; SI: s_endpgm
define amdgpu_kernel void @s_addk_v8i32_k0(<8 x i32> addrspace(1)* %out, <8 x i32> %b) {
  %add = add <8 x i32> %b, <i32 65, i32 66, i32 67, i32 68, i32 69, i32 70, i32 71, i32 72>
  store <8 x i32> %add, <8 x i32> addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}no_s_addk_i32_k0:
; SI: s_add_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x8000{{$}}
; SI: s_endpgm
define amdgpu_kernel void @no_s_addk_i32_k0(i32 addrspace(1)* %out, i32 %b) {
  %add = add i32 %b, 32768 ; 1 << 15
  store i32 %add, i32 addrspace(1)* %out
  ret void
}

@lds = addrspace(3) global [512 x i32] undef, align 4

; SI-LABEL: {{^}}commute_s_addk_i32:
; SI: s_addk_i32 s{{[0-9]+}}, 0x800{{$}}
define amdgpu_kernel void @commute_s_addk_i32(i32 addrspace(1)* %out, i32 %b) #0 {
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %add = add i32 %size, %b
  call void asm sideeffect "; foo $0, $1", "v,s"([512 x i32] addrspace(3)* @lds, i32 %add)
  ret void
}

declare i32 @llvm.amdgcn.groupstaticsize() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
