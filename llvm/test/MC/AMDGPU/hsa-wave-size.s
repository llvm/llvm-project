// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=2 %s | FileCheck --check-prefixes=GCN,GFX7 %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1010 --amdhsa-code-object-version=2 -mattr=+wavefrontsize32,-wavefrontsize64 %s | FileCheck --check-prefixes=GCN,GFX10-W32 %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1010 --amdhsa-code-object-version=2 -mattr=-wavefrontsize32,+wavefrontsize64 %s | FileCheck --check-prefixes=GCN,GFX10-W64 %s

// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 --amdhsa-code-object-version=2 %s 2>&1 | FileCheck --check-prefix=GFX7-ERR %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1010 --amdhsa-code-object-version=2 -mattr=+wavefrontsize32,-wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=GFX10-W32-ERR %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1010 --amdhsa-code-object-version=2 -mattr=-wavefrontsize32,+wavefrontsize64 %s 2>&1 | FileCheck --check-prefix=GFX10-W64-ERR %s

// GCN: test0:
// GFX7: enable_wavefront_size32 = 0
// GFX7: wavefront_size = 6
// GFX10-W32: enable_wavefront_size32 = 1
// GFX10-W32: wavefront_size = 5
// GFX10-W64: enable_wavefront_size32 = 0
// GFX10-W64: wavefront_size = 6
.amdgpu_hsa_kernel test0
test0:
.amd_kernel_code_t
.end_amd_kernel_code_t

// GCN: test1:
// GFX7: enable_wavefront_size32 = 0
// GFX7: wavefront_size = 6
// GFX10-W32-ERR: error: enable_wavefront_size32=0 requires +WavefrontSize64
// GFX10-W64: enable_wavefront_size32 = 0
// GFX10-W64: wavefront_size = 6
.amdgpu_hsa_kernel test1
test1:
.amd_kernel_code_t
  enable_wavefront_size32 = 0
.end_amd_kernel_code_t

// GCN: test2:
// GFX7: enable_wavefront_size32 = 0
// GFX7: wavefront_size = 6
// GFX10-W32-ERR: error: wavefront_size=6 requires +WavefrontSize64
// GFX10-W64: enable_wavefront_size32 = 0
// GFX10-W64: wavefront_size = 6
.amdgpu_hsa_kernel test2
test2:
.amd_kernel_code_t
  wavefront_size = 6
.end_amd_kernel_code_t

// GCN: test3:
// GFX7-ERR: error: enable_wavefront_size32=1 is only allowed on GFX10+
// GFX10-W32: enable_wavefront_size32 = 1
// GFX10-W32: wavefront_size = 5
// GFX10-W64-ERR: error: enable_wavefront_size32=1 requires +WavefrontSize32
.amdgpu_hsa_kernel test3
test3:
.amd_kernel_code_t
  enable_wavefront_size32 = 1
.end_amd_kernel_code_t

// GCN: test4:
// GFX7-ERR: error: wavefront_size=5 is only allowed on GFX10+
// GFX10-W32: enable_wavefront_size32 = 1
// GFX10-W32: wavefront_size = 5
// GFX10-W64-ERR: error: wavefront_size=5 requires +WavefrontSize32
.amdgpu_hsa_kernel test4
test4:
.amd_kernel_code_t
  wavefront_size = 5
.end_amd_kernel_code_t
