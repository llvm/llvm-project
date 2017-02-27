// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=GFX700
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=GFX800
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -show-encoding %s | FileCheck %s --check-prefix=GCN --check-prefix=GFX900

.amdgpu_runtime_metadata
  { amd.MDVersion: [ 2, 1 ], amd.IsaInfo: { amd.IsaInfoWavefrontSize: 64, amd.IsaInfoLocalMemorySize: 65536, amd.IsaInfoEUsPerCU: 4, amd.IsaInfoMaxWavesPerEU: 10, amd.IsaInfoMaxFlatWorkGroupSize: 2048, amd.IsaInfoSGPRAllocGranule: 8, amd.IsaInfoTotalNumSGPRs: 512, amd.IsaInfoAddressableNumSGPRs: 104, amd.IsaInfoVGPRAllocGranule: 4, amd.IsaInfoTotalNumVGPRs: 256, amd.IsaInfoAddressableNumVGPRs: 256 }, amd.PrintfInfo: [ '1:1:4:%d\n', '2:1:8:%g\n' ], amd.Kernels:
    - { amd.KernelName: test_char, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args:
        - { amd.ArgSize: 1, amd.ArgAlign: 1, amd.ArgKind: 0, amd.ArgValueType: 1, amd.ArgTypeName: char, amd.ArgAccQual: 0 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
    - { amd.KernelName: test_ushort2, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args:
        - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 4, amd.ArgTypeName: ushort2, amd.ArgAccQual: 0 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
        - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
  }
.end_amdgpu_runtime_metadata

// GFX700: { amd.MDVersion: [ 2, 1 ], amd.IsaInfo: { amd.IsaInfoWavefrontSize: 64, amd.IsaInfoLocalMemorySize: 65536, amd.IsaInfoEUsPerCU: 4, amd.IsaInfoMaxWavesPerEU: 10, amd.IsaInfoMaxFlatWorkGroupSize: 2048, amd.IsaInfoSGPRAllocGranule: 8, amd.IsaInfoTotalNumSGPRs: 512, amd.IsaInfoAddressableNumSGPRs: 104, amd.IsaInfoVGPRAllocGranule: 4, amd.IsaInfoTotalNumVGPRs: 256, amd.IsaInfoAddressableNumVGPRs: 256 }, amd.PrintfInfo: [ '1:1:4:%d\n', '2:1:8:%g\n' ], amd.Kernels:

// GFX800: { amd.MDVersion: [ 2, 1 ], amd.IsaInfo: { amd.IsaInfoWavefrontSize: 64, amd.IsaInfoLocalMemorySize: 65536, amd.IsaInfoEUsPerCU: 4, amd.IsaInfoMaxWavesPerEU: 10, amd.IsaInfoMaxFlatWorkGroupSize: 2048, amd.IsaInfoSGPRAllocGranule: 16, amd.IsaInfoTotalNumSGPRs: 800, amd.IsaInfoAddressableNumSGPRs: 96, amd.IsaInfoVGPRAllocGranule: 4, amd.IsaInfoTotalNumVGPRs: 256, amd.IsaInfoAddressableNumVGPRs: 256 }, amd.PrintfInfo: [ '1:1:4:%d\n', '2:1:8:%g\n' ], amd.Kernels:

// GFX900: { amd.MDVersion: [ 2, 1 ], amd.IsaInfo: { amd.IsaInfoWavefrontSize: 64, amd.IsaInfoLocalMemorySize: 65536, amd.IsaInfoEUsPerCU: 4, amd.IsaInfoMaxWavesPerEU: 10, amd.IsaInfoMaxFlatWorkGroupSize: 2048, amd.IsaInfoSGPRAllocGranule: 16, amd.IsaInfoTotalNumSGPRs: 800, amd.IsaInfoAddressableNumSGPRs: 102, amd.IsaInfoVGPRAllocGranule: 4, amd.IsaInfoTotalNumVGPRs: 256, amd.IsaInfoAddressableNumVGPRs: 256 }, amd.PrintfInfo: [ '1:1:4:%d\n', '2:1:8:%g\n' ], amd.Kernels:

// GCN:      - { amd.KernelName: test_char, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args:
// GCN-NEXT:     - { amd.ArgSize: 1, amd.ArgAlign: 1, amd.ArgKind: 0, amd.ArgValueType: 1, amd.ArgTypeName: char, amd.ArgAccQual: 0 }
// GCN-NEXT:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
// GCN-NEXT:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
// GCN-NEXT:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
// GCN-NEXT:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
// GCN-NEXT: - { amd.KernelName: test_ushort2, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args:
// GCN-NEXT:     - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 4, amd.ArgTypeName: ushort2, amd.ArgAccQual: 0 }
// GCN-NEXT:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
// GCN-NEXT:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
// GCN-NEXT:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
// GCN-NEXT:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } } }
