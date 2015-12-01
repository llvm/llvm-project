; RUN: llc -march=amdgcn -mcpu=SI -verify-machineinstrs < %s | FileCheck -check-prefix=SI -check-prefix=GCN -check-prefix=SI-NOHSA -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=amdgcn -mcpu=tonga -verify-machineinstrs < %s | FileCheck -check-prefix=VI  -check-prefix=VI-NOHSA -check-prefix=GCN -check-prefix=GCN-NOHSA -check-prefix=FUNC %s
; RUN: llc -march=r600 -mcpu=redwood < %s | FileCheck -check-prefix=EG -check-prefix=FUNC %s

; FUNC-LABEL: {{^}}read_workdim:
; EG: MEM_RAT_CACHELESS STORE_RAW [[VAL:T[0-9]+\.X]]
; EG: MOV [[VAL]], KC0[2].Z

; SI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0xb
; VI-NOHSA: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x2c
; GCN-NOHSA: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN-NOHSA: buffer_store_dword [[VVAL]]
define void @read_workdim(i32 addrspace(1)* %out) {
entry:
  %0 = call i32 @llvm.AMDGPU.read.workdim() #0
  store i32 %0, i32 addrspace(1)* %out
  ret void
}

; FUNC-LABEL: {{^}}read_workdim_known_bits:
; SI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0xb
; VI: s_load_dword [[VAL:s[0-9]+]], s[0:1], 0x2c
; GCN-NOT: 0xff
; GCN: v_mov_b32_e32 [[VVAL:v[0-9]+]], [[VAL]]
; GCN: buffer_store_dword [[VVAL]]
define void @read_workdim_known_bits(i32 addrspace(1)* %out) {
entry:
  %dim = call i32 @llvm.AMDGPU.read.workdim() #0
  %shl = shl i32 %dim, 24
  %shr = lshr i32 %shl, 24
  store i32 %shr, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.AMDGPU.read.workdim() #0

attributes #0 = { readnone }
