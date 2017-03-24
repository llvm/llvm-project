// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -filetype=obj %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -filetype=obj %s 2>&1 | FileCheck %s
// RUN: not llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -filetype=obj %s 2>&1 | FileCheck %s

// CHECK: error: unknown key 'UnknownKey'
.amdgpu_code_object_metadata
  UnknownKey: [ 2, 0 ]
  Version: [ 1, 0 ]
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
