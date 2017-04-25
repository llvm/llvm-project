; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI -check-prefix=OPT %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=hawaii -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=CI -check-prefix=OPT %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=iceland -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=VI -check-prefix=OPT %s
; RUN: llc -O0 -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -enable-var-scope -check-prefix=GCN -check-prefix=OPTNONE %s

; There are no stack objects, but still a private memory access. The
; private access regiters need to be correctly initialized anyway, and
; shifted down to the end of the used registers.

; GCN-LABEL: {{^}}store_to_undef:
; OPT-DAG: s_mov_b64 s{{\[}}[[RSRC_LO:[0-9]+]]:{{[0-9]+\]}}, s[0:1]
; OPT-DAG: s_mov_b64 s{{\[[0-9]+}}:[[RSRC_HI:[0-9]+]]{{\]}}, s[2:3]
; OPT-DAG: s_mov_b32 [[SOFFSET:s[0-9]+]], s7{{$}}
; OPT: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[}}[[RSRC_LO]]:[[RSRC_HI]]{{\]}}, [[SOFFSET]] offen{{$}}

; -O0 should assume spilling, so the input scratch resource descriptor
; -should be used directly without any copies.

; OPTNONE-NOT: s_mov_b32
; OPTNONE: buffer_store_dword v{{[0-9]+}}, v{{[0-9]+}}, s[0:3], s7 offen{{$}}
define amdgpu_kernel void @store_to_undef() #0 {
  store volatile i32 0, i32* undef
  ret void
}

; GCN-LABEL: {{^}}store_to_inttoptr:
; OPT-DAG: s_mov_b64 s{{\[}}[[RSRC_LO:[0-9]+]]:{{[0-9]+\]}}, s[0:1]
; OPT-DAG: s_mov_b64 s{{\[[0-9]+}}:[[RSRC_HI:[0-9]+]]{{\]}}, s[2:3]
; OPT-DAG: s_mov_b32 [[SOFFSET:s[0-9]+]], s7{{$}}
; OPT: buffer_store_dword v{{[0-9]+}}, off, s{{\[}}[[RSRC_LO]]:[[RSRC_HI]]{{\]}}, [[SOFFSET]] offset:124{{$}}
define amdgpu_kernel void @store_to_inttoptr() #0 {
 store volatile i32 0, i32* inttoptr (i32 124 to i32*)
 ret void
}

; GCN-LABEL: {{^}}load_from_undef:
; OPT-DAG: s_mov_b64 s{{\[}}[[RSRC_LO:[0-9]+]]:{{[0-9]+\]}}, s[0:1]
; OPT-DAG: s_mov_b64 s{{\[[0-9]+}}:[[RSRC_HI:[0-9]+]]{{\]}}, s[2:3]
; OPT-DAG: s_mov_b32 [[SOFFSET:s[0-9]+]], s7{{$}}
; OPT: buffer_load_dword v{{[0-9]+}}, v{{[0-9]+}}, s{{\[}}[[RSRC_LO]]:[[RSRC_HI]]{{\]}}, [[SOFFSET]] offen{{$}}
define amdgpu_kernel void @load_from_undef() #0 {
  %ld = load volatile i32, i32* undef
  ret void
}

; GCN-LABEL: {{^}}load_from_inttoptr:
; OPT-DAG: s_mov_b64 s{{\[}}[[RSRC_LO:[0-9]+]]:{{[0-9]+\]}}, s[0:1]
; OPT-DAG: s_mov_b64 s{{\[[0-9]+}}:[[RSRC_HI:[0-9]+]]{{\]}}, s[2:3]
; OPT-DAG: s_mov_b32 [[SOFFSET:s[0-9]+]], s7{{$}}
; OPT: buffer_load_dword v{{[0-9]+}}, off, s{{\[}}[[RSRC_LO]]:[[RSRC_HI]]{{\]}}, [[SOFFSET]] offset:124{{$}}
define amdgpu_kernel void @load_from_inttoptr() #0 {
  %ld = load volatile i32, i32* inttoptr (i32 124 to i32*)
  ret void
}

attributes #0 = { nounwind }
