// RUN: not llvm-mc -arch amdgcn %s 2>&1 | FileCheck --check-prefix=GCN %s

// GCN: error: .amd_amdgpu_hsa_metadata directive is not available on non-amdhsa OSes
.amd_amdgpu_hsa_metadata

// GCN: error: .amd_amdgpu_pal_metadata directive is not available on non-amdpal OSes
.amd_amdgpu_pal_metadata
