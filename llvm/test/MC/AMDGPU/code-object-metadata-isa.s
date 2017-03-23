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
// CHECK:        Args:
// CHECK:          - Size:          1
// CHECK:            Align:         1
// CHECK:            Kind:          ByValue
// CHECK:            ValueType:     I8
// CHECK:            AccQual:       Default
// CHECK:            TypeName:      char
// CHECK:          - Size:          8
// CHECK:            Align:         8
// CHECK:            Kind:          HiddenGlobalOffsetX
// CHECK:            ValueType:     I64
// CHECK:          - Size:          8
// CHECK:            Align:         8
// CHECK:            Kind:          HiddenGlobalOffsetY
// CHECK:            ValueType:     I64
// CHECK:          - Size:          8
// CHECK:            Align:         8
// CHECK:            Kind:          HiddenGlobalOffsetZ
// CHECK:            ValueType:     I64
// CHECK:          - Size:          8
// CHECK:            Align:         8
// CHECK:            Kind:          HiddenPrintfBuffer
// CHECK:            ValueType:     I8
// CHECK:            AddrSpaceQual: Global
// CHECK:  .end_amdgpu_code_object_metadata
.amdgpu_code_object_metadata
  Version: [ 1, 0 ]
  Isa:
    WavefrontSize:        1
    LocalMemorySize:      1
    EUsPerCU:             1
    MaxWavesPerEU:        1
    MaxFlatWorkGroupSize: 1
    SGPRAllocGranule:     1
    TotalNumSGPRs:        1
    AddressableNumSGPRs:  1
    VGPRAllocGranule:     1
    TotalNumVGPRs:        1
    AddressableNumVGPRs:  1
  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
  Kernels:
    - Name:            test_kernel
      Language:        OpenCL C
      LanguageVersion: [ 2, 0 ]
      Args:
        - Size:          1
          Align:         1
          Kind:          ByValue
          ValueType:     I8
          AccQual:       Default
          TypeName:      char
        - Size:          8
          Align:         8
          Kind:          HiddenGlobalOffsetX
          ValueType:     I64
        - Size:          8
          Align:         8
          Kind:          HiddenGlobalOffsetY
          ValueType:     I64
        - Size:          8
          Align:         8
          Kind:          HiddenGlobalOffsetZ
          ValueType:     I64
        - Size:          8
          Align:         8
          Kind:          HiddenPrintfBuffer
          ValueType:     I8
          AddrSpaceQual: Global
.end_amdgpu_code_object_metadata
