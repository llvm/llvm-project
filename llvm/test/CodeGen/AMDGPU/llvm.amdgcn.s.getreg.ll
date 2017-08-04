; RUN: llc -march=amdgcn -mcpu=tahiti -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=kaveri -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgcn--amdhsa -mcpu=fiji -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; GCN-LABEL: {{^}}s_getreg_test:
; GCN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_LDS_ALLOC, 8, 23)
define amdgpu_kernel void @s_getreg_test(i32 addrspace(1)* %out) { ; simm16=45574 for lds size.
  %lds_size_64dwords = call i32 @llvm.amdgcn.s.getreg(i32 45574)
  %lds_size_bytes = shl i32 %lds_size_64dwords, 8
  store i32 %lds_size_bytes, i32 addrspace(1)* %out
  ret void
}

; Call site has additional readnone knowledge.
; GCN-LABEL: {{^}}readnone_s_getreg_test:
; GCN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_LDS_ALLOC, 8, 23)
define amdgpu_kernel void @readnone_s_getreg_test(i32 addrspace(1)* %out) { ; simm16=45574 for lds size.
  %lds_size_64dwords = call i32 @llvm.amdgcn.s.getreg(i32 45574) #1
  %lds_size_bytes = shl i32 %lds_size_64dwords, 8
  store i32 %lds_size_bytes, i32 addrspace(1)* %out
  ret void
}

declare i32 @llvm.amdgcn.s.getreg(i32) #0

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind readnone }
