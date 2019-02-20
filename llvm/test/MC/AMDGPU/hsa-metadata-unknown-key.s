// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=-code-object-v3 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -mattr=-code-object-v3 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -mattr=-code-object-v3 -filetype=obj %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -mattr=-code-object-v3 -filetype=obj %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -mattr=-code-object-v3 -filetype=obj %s 2>&1 | FileCheck %s

// CHECK: error: unknown key 'UnknownKey'
.amd_amdgpu_hsa_metadata
  UnknownKey: [ 2, 0 ]
  Version: [ 1, 0 ]
  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
  Kernels:
    - Name:            test_kernel
      SymbolName:      test_kernel@kd
      Language:        OpenCL C
      LanguageVersion: [ 2, 0 ]
      Args:
        - Size:          1
          Align:         1
          ValueKind:     ByValue
          ValueType:     I8
          AccQual:       Default
          TypeName:      char
        - Size:          8
          Align:         8
          ValueKind:     HiddenGlobalOffsetX
          ValueType:     I64
        - Size:          8
          Align:         8
          ValueKind:     HiddenGlobalOffsetY
          ValueType:     I64
        - Size:          8
          Align:         8
          ValueKind:     HiddenGlobalOffsetZ
          ValueType:     I64
        - Size:          8
          Align:         8
          ValueKind:     HiddenPrintfBuffer
          ValueType:     I8
          AddrSpaceQual: Global
.end_amd_amdgpu_hsa_metadata
