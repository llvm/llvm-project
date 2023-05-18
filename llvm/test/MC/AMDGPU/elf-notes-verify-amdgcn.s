// RUN: not llvm-mc -triple=amdgcn %s 2>&1 | FileCheck --check-prefix=GCN %s

// GCN: :[[@LINE+1]]:{{[0-9]+}}: error: .amd_amdgpu_hsa_metadata directive is not available on non-amdhsa OSes
.amd_amdgpu_hsa_metadata

// GCN: :[[@LINE+1]]:{{[0-9]+}}: error: .amd_amdgpu_pal_metadata directive is not available on non-amdpal OSes
.amd_amdgpu_pal_metadata
