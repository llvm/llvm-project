; RUN: llc -mtriple=amdgpu6.00 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu7.00--amdhsa < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -mtriple=amdgpu8.03--amdhsa < %s | FileCheck -check-prefix=GCN %s

; RUN: llc -global-isel -mtriple=amdgpu6.00 < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel -mtriple=amdgpu7.00--amdhsa < %s | FileCheck -check-prefix=GCN %s
; RUN: llc -global-isel -mtriple=amdgpu8.03--amdhsa < %s | FileCheck -check-prefix=GCN %s


; GCN-LABEL: {{^}}s_getreg_test:
; GCN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_LDS_ALLOC, 8, 23)
define amdgpu_kernel void @s_getreg_test(ptr addrspace(1) %out) { ; simm16=45574 for lds size.
  %lds_size_64dwords = call i32 @llvm.amdgcn.s.getreg(i32 45574)
  %lds_size_bytes = shl i32 %lds_size_64dwords, 8
  store i32 %lds_size_bytes, ptr addrspace(1) %out
  ret void
}

; Call site has additional readnone knowledge.
; GCN-LABEL: {{^}}readnone_s_getreg_test:
; GCN: s_getreg_b32 s{{[0-9]+}}, hwreg(HW_REG_LDS_ALLOC, 8, 23)
define amdgpu_kernel void @readnone_s_getreg_test(ptr addrspace(1) %out) { ; simm16=45574 for lds size.
  %lds_size_64dwords = call i32 @llvm.amdgcn.s.getreg(i32 45574) #1
  %lds_size_bytes = shl i32 %lds_size_64dwords, 8
  store i32 %lds_size_bytes, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.s.getreg(i32 immarg) #0

attributes #0 = { nounwind readonly }
attributes #1 = { nounwind readnone }
