; RUN: llc -march=amdgcn -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s
; RUN: llc -march=amdgcn -mcpu=tonga -mattr=-flat-for-global -verify-machineinstrs < %s | FileCheck -check-prefix=SI %s

; SI-LABEL: {{^}}s_mulk_i32_k0:
; SI: s_load_dword [[VAL:s[0-9]+]]
; SI: s_mulk_i32 [[VAL]], 0x41
; SI: v_mov_b32_e32 [[VRESULT:v[0-9]+]], [[VAL]]
; SI: buffer_store_dword [[VRESULT]]
; SI: s_endpgm
define void @s_mulk_i32_k0(i32 addrspace(1)* %out, i32 %b) {
  %mul = mul i32 %b, 65
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_mulk_i32_k1:
; SI: s_mulk_i32 {{s[0-9]+}}, 0x7fff{{$}}
; SI: s_endpgm
define void @s_mulk_i32_k1(i32 addrspace(1)* %out, i32 %b) {
  %mul = mul i32 %b, 32767 ; (1 << 15) - 1
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}s_mulk_i32_k2:
; SI: s_mulk_i32 {{s[0-9]+}}, 0xffef{{$}}
; SI: s_endpgm
define void @s_mulk_i32_k2(i32 addrspace(1)* %out, i32 %b) {
  %mul = mul i32 %b, -17
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

; SI-LABEL: {{^}}no_s_mulk_i32_k0:
; SI: s_mul_i32 {{s[0-9]+}}, {{s[0-9]+}}, 0x8001{{$}}
; SI: s_endpgm
define void @no_s_mulk_i32_k0(i32 addrspace(1)* %out, i32 %b) {
  %mul = mul i32 %b, 32769 ; 1 << 15 + 1
  store i32 %mul, i32 addrspace(1)* %out
  ret void
}

@lds = addrspace(3) global [512 x i32] undef, align 4

; SI-LABEL: {{^}}commute_s_mulk_i32:
; SI: s_mulk_i32 s{{[0-9]+}}, 0x800{{$}}
define void @commute_s_mulk_i32(i32 addrspace(1)* %out, i32 %b) #0 {
  %size = call i32 @llvm.amdgcn.groupstaticsize()
  %add = mul i32 %size, %b
  call void asm sideeffect "; foo $0, $1", "v,s"([512 x i32] addrspace(3)* @lds, i32 %add)
  ret void
}

declare i32 @llvm.amdgcn.groupstaticsize() #1

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }
