// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX700 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX800 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX900 %s

// CHECK:  .amdgpu_code_object_metadata
// CHECK:    Version: [ 1, 0 ]
// CHECK:    Kernels:
// CHECK:      - Name: test_kernel
// CHECK:        CodeProps:
// CHECK:          KernargSegmentSize:         24
// CHECK:          WorkitemPrivateSegmentSize: 16
// CHECK:          WavefrontNumSGPRs:          6
// CHECK:          WorkitemNumVGPRs:           12
.amdgpu_code_object_metadata
  Version: [ 1, 0 ]
  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
  Kernels:
    - Name:            test_kernel
      CodeProps:
        KernargSegmentSize:         24
        WorkitemPrivateSegmentSize: 16
        WavefrontNumSGPRs:          6
        WorkitemNumVGPRs:           12
.end_amdgpu_code_object_metadata
