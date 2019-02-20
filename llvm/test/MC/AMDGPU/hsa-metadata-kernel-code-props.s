// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=-code-object-v3 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX700 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -mattr=-code-object-v3 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX800 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX900 %s

// CHECK:  .amd_amdgpu_hsa_metadata
// CHECK:    Version: [ 1, 0 ]
// CHECK:    Kernels:
// CHECK:      - Name:       test_kernel
// CHECK:        SymbolName: 'test_kernel@kd'
// CHECK:        CodeProps:
// CHECK:          KernargSegmentSize:      24
// CHECK:          GroupSegmentFixedSize:   24
// CHECK:          PrivateSegmentFixedSize: 16
// CHECK:          KernargSegmentAlign:     16
// CHECK:          WavefrontSize:           64
// CHECK:          MaxFlatWorkGroupSize:    256
// CHECK:          NumSpilledSGPRs: 1
// CHECK:          NumSpilledVGPRs: 1
.amd_amdgpu_hsa_metadata
  Version: [ 1, 0 ]
  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
  Kernels:
    - Name:            test_kernel
      SymbolName:      test_kernel@kd
      CodeProps:
        KernargSegmentSize:      24
        GroupSegmentFixedSize:   24
        PrivateSegmentFixedSize: 16
        KernargSegmentAlign:     16
        WavefrontSize:           64
        MaxFlatWorkGroupSize:    256
        NumSpilledSGPRs:         1
        NumSpilledVGPRs:         1
.end_amd_amdgpu_hsa_metadata
