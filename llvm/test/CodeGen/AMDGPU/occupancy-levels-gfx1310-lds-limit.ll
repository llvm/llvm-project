; RUN: llc -mtriple=amdgpu13.10 < %s | FileCheck --check-prefixes=GCN,W32WG-LDS192 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+wavefrontsize64 < %s | FileCheck --check-prefixes=GCN,W64WG-LDS192 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+cumode < %s | FileCheck --check-prefixes=GCN,W32CU-LDS192 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+wavefrontsize64,+cumode < %s | FileCheck --check-prefixes=GCN,W64CU-LDS192 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+local-memory-size-limit-131072 < %s | FileCheck --check-prefixes=GCN,W32WG-LDS128 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+wavefrontsize64,+local-memory-size-limit-131072 < %s | FileCheck --check-prefixes=GCN,W64WG-LDS128 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+cumode,+local-memory-size-limit-131072 < %s | FileCheck --check-prefixes=GCN,W32CU-LDS128 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+wavefrontsize64,+cumode,+local-memory-size-limit-131072 < %s | FileCheck --check-prefixes=GCN,W64CU-LDS128 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+local-memory-size-limit-65536 < %s | FileCheck --check-prefixes=GCN,W32WG-LDS64 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+wavefrontsize64,+local-memory-size-limit-65536 < %s | FileCheck --check-prefixes=GCN,W64WG-LDS64 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+cumode,+local-memory-size-limit-65536 < %s | FileCheck --check-prefixes=GCN,W32CU-LDS64 %s
; RUN: llc -mtriple=amdgpu13.10 -mattr=+wavefrontsize64,+cumode,+local-memory-size-limit-65536 < %s | FileCheck --check-prefixes=GCN,W64CU-LDS64 %s

@lds8252 = internal addrspace(3) global [8252 x i8] poison, align 4

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_32:
; W32WG-LDS192: ; Occupancy: 6{{$}}
; W32CU-LDS192: ; Occupancy: 5{{$}}
; W64WG-LDS192: ; Occupancy: 6{{$}}
; W64CU-LDS192: ; Occupancy: 5{{$}}
; W32WG-LDS128: ; Occupancy: 4{{$}}
; W32CU-LDS128: ; Occupancy: 4{{$}}
; W64WG-LDS128: ; Occupancy: 4{{$}}
; W64CU-LDS128: ; Occupancy: 4{{$}}
; W32WG-LDS64: ; Occupancy: 2{{$}}
; W32CU-LDS64: ; Occupancy: 2{{$}}
; W64WG-LDS64: ; Occupancy: 2{{$}}
; W64CU-LDS64: ; Occupancy: 2{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_32() #0 {
  store volatile i8 1, ptr addrspace(3) @lds8252
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_64:
; W32WG-LDS192: ; Occupancy: 11{{$}}
; W32CU-LDS192: ; Occupancy: 10{{$}}
; W64WG-LDS192: ; Occupancy: 6{{$}}
; W64CU-LDS192: ; Occupancy: 5{{$}}
; W32WG-LDS128: ; Occupancy: 7{{$}}
; W32CU-LDS128: ; Occupancy: 7{{$}}
; W64WG-LDS128: ; Occupancy: 4{{$}}
; W64CU-LDS128: ; Occupancy: 4{{$}}
; W32WG-LDS64: ; Occupancy: 4{{$}}
; W32CU-LDS64: ; Occupancy: 3{{$}}
; W64WG-LDS64: ; Occupancy: 2{{$}}
; W64CU-LDS64: ; Occupancy: 2{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_64() #1 {
  store volatile i8 1, ptr addrspace(3) @lds8252
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_96:
; W32WG-LDS192: ; Occupancy: 16{{$}}
; W32CU-LDS192: ; Occupancy: 15{{$}}
; W64WG-LDS192: ; Occupancy: 11{{$}}
; W64CU-LDS192: ; Occupancy: 10{{$}}
; W32WG-LDS128: ; Occupancy: 11{{$}}
; W32CU-LDS128: ; Occupancy: 11{{$}}
; W64WG-LDS128: ; Occupancy: 7{{$}}
; W64CU-LDS128: ; Occupancy: 7{{$}}
; W32WG-LDS64: ; Occupancy: 6{{$}}
; W32CU-LDS64: ; Occupancy: 5{{$}}
; W64WG-LDS64: ; Occupancy: 4{{$}}
; W64CU-LDS64: ; Occupancy: 3{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_96() #2 {
  store volatile i8 1, ptr addrspace(3) @lds8252
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_128:
; W32WG-LDS192: ; Occupancy: 16{{$}}
; W32CU-LDS192: ; Occupancy: 16{{$}}
; W64WG-LDS192: ; Occupancy: 11{{$}}
; W64CU-LDS192: ; Occupancy: 10{{$}}
; W32WG-LDS128: ; Occupancy: 14{{$}}
; W32CU-LDS128: ; Occupancy: 14{{$}}
; W64WG-LDS128: ; Occupancy: 7{{$}}
; W64CU-LDS128: ; Occupancy: 7{{$}}
; W32WG-LDS64: ; Occupancy: 7{{$}}
; W32CU-LDS64: ; Occupancy: 6{{$}}
; W64WG-LDS64: ; Occupancy: 4{{$}}
; W64CU-LDS64: ; Occupancy: 3{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_128() #3 {
  store volatile i8 1, ptr addrspace(3) @lds8252
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_192:
; W32WG-LDS192: ; Occupancy: 15{{$}}
; W32CU-LDS192: ; Occupancy: 15{{$}}
; W64WG-LDS192: ; Occupancy: 16{{$}}
; W64CU-LDS192: ; Occupancy: 15{{$}}
; W32WG-LDS128: ; Occupancy: 15{{$}}
; W32CU-LDS128: ; Occupancy: 15{{$}}
; W64WG-LDS128: ; Occupancy: 11{{$}}
; W64CU-LDS128: ; Occupancy: 11{{$}}
; W32WG-LDS64: ; Occupancy: 11{{$}}
; W32CU-LDS64: ; Occupancy: 9{{$}}
; W64WG-LDS64: ; Occupancy: 6{{$}}
; W64CU-LDS64: ; Occupancy: 5{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_192() #4 {
  store volatile i8 1, ptr addrspace(3) @lds8252
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_256:
; W32WG-LDS192: ; Occupancy: 16{{$}}
; W32CU-LDS192: ; Occupancy: 16{{$}}
; W64WG-LDS192: ; Occupancy: 16{{$}}
; W64CU-LDS192: ; Occupancy: 16{{$}}
; W32WG-LDS128: ; Occupancy: 16{{$}}
; W32CU-LDS128: ; Occupancy: 16{{$}}
; W64WG-LDS128: ; Occupancy: 14{{$}}
; W64CU-LDS128: ; Occupancy: 14{{$}}
; W32WG-LDS64: ; Occupancy: 14{{$}}
; W32CU-LDS64: ; Occupancy: 12{{$}}
; W64WG-LDS64: ; Occupancy: 7{{$}}
; W64CU-LDS64: ; Occupancy: 6{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_256() #5 {
  store volatile i8 1, ptr addrspace(3) @lds8252
  ret void
}

; GCN-LABEL: {{^}}used_lds_8252_max_group_size_512:
; W32WG-LDS192: ; Occupancy: 16{{$}}
; W32CU-LDS192: ; Occupancy: 16{{$}}
; W64WG-LDS192: ; Occupancy: 16{{$}}
; W64CU-LDS192: ; Occupancy: 16{{$}}
; W32WG-LDS128: ; Occupancy: 16{{$}}
; W32CU-LDS128: ; Occupancy: 16{{$}}
; W64WG-LDS128: ; Occupancy: 16{{$}}
; W64CU-LDS128: ; Occupancy: 16{{$}}
; W32WG-LDS64: ; Occupancy: 16{{$}}
; W32CU-LDS64: ; Occupancy: 16{{$}}
; W64WG-LDS64: ; Occupancy: 14{{$}}
; W64CU-LDS64: ; Occupancy: 12{{$}}
define amdgpu_kernel void @used_lds_8252_max_group_size_512() #6 {
  store volatile i8 1, ptr addrspace(3) @lds8252
  ret void
}

attributes #0 = { "amdgpu-flat-work-group-size"="1,32" }
attributes #1 = { "amdgpu-flat-work-group-size"="1,64" }
attributes #2 = { "amdgpu-flat-work-group-size"="1,96" }
attributes #3 = { "amdgpu-flat-work-group-size"="1,128" }
attributes #4 = { "amdgpu-flat-work-group-size"="1,192" }
attributes #5 = { "amdgpu-flat-work-group-size"="1,256" }
attributes #6 = { "amdgpu-flat-work-group-size"="1,512" }
