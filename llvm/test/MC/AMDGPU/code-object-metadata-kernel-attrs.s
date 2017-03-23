// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX700 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX800 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX900 %s

// CHECK:  .amdgpu_code_object_metadata
// CHECK:    Version: [ 1, 0 ]
// CHECK:    Isa:
// CHECK:      WavefrontSize:        64
// CHECK:      LocalMemorySize:      65536
// CHECK:      EUsPerCU:             4
// CHECK:      MaxWavesPerEU:        10
// CHECK:      MaxFlatWorkGroupSize: 2048
// GFX700:     SGPRAllocGranule:     8
// GFX800:     SGPRAllocGranule:     16
// GFX900:     SGPRAllocGranule:     16
// GFX700:     TotalNumSGPRs:        512
// GFX800:     TotalNumSGPRs:        800
// GFX900:     TotalNumSGPRs:        800
// GFX700:     AddressableNumSGPRs:  104
// GFX800:     AddressableNumSGPRs:  96
// GFX900:     AddressableNumSGPRs:  102
// CHECK:      VGPRAllocGranule:     4
// CHECK:      TotalNumVGPRs:        256
// CHECK:      AddressableNumVGPRs:  256
// CHECK:    Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
// CHECK:    Kernels:
// CHECK:      - Name:            test_kernel
// CHECK:        Language:        OpenCL C
// CHECK:        LanguageVersion: [ 2, 0 ]
// CHECK:    Attrs:
// CHECK:        ReqdWorkGroupSize: [ 1, 2, 4 ]
// CHECK:        WorkGroupSizeHint: [ 8, 16, 32 ]
// CHECK:        VecTypeHint:       int
// CHECK: .end_amdgpu_code_object_metadata
.amdgpu_code_object_metadata
  Version: [ 1, 0 ]
  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
  Kernels:
    - Name:            test_kernel
      Language:        OpenCL C
      LanguageVersion: [ 2, 0 ]
      Attrs:
        ReqdWorkGroupSize: [ 1, 2, 4 ]
        WorkGroupSizeHint: [ 8, 16, 32 ]
        VecTypeHint:       int
.end_amdgpu_code_object_metadata
