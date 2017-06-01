// RUN: llvm-mc -triple amdgcn--amdhsa -mcpu=kaveri -show-encoding %s | FileCheck %s --check-prefix=ASM

.amdgpu_runtime_metadata
    { amd.MDVersion: [ 2, 0 ], amd.PrintfInfo: [ '1:1:4:%d\n', '2:1:8:%g\n' ], amd.Kernels:

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

// ASM: { amd.MDVersion: [ 2, 0 ], amd.PrintfInfo: [ '1:1:4:%d\n', '2:1:8:%g\n' ], amd.Kernels:
// ASM: - { amd.KernelName: test_char, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args:
// ASM:     - { amd.ArgSize: 1, amd.ArgAlign: 1, amd.ArgKind: 0, amd.ArgValueType: 1, amd.ArgTypeName: char, amd.ArgAccQual: 0 }
// ASM:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
// ASM:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
// ASM:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
// ASM:     - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
// ASM: - { amd.KernelName: test_ushort2, amd.Language: OpenCL C, amd.LanguageVersion: [ 2, 0 ], amd.Args:
// ASM:    - { amd.ArgSize: 4, amd.ArgAlign: 4, amd.ArgKind: 0, amd.ArgValueType: 4, amd.ArgTypeName: ushort2, amd.ArgAccQual: 0 }
// ASM:    - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 7, amd.ArgValueType: 9 }
// ASM:    - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 8, amd.ArgValueType: 9 }
// ASM:    - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 9, amd.ArgValueType: 9 }
// ASM:    - { amd.ArgSize: 8, amd.ArgAlign: 8, amd.ArgKind: 11, amd.ArgValueType: 1, amd.ArgAddrQual: 1 } }
// ASM: }
