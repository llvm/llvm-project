// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx700 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX700 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx800 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX800 %s
// RUN: llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx900 -show-encoding %s | FileCheck --check-prefix=CHECK --check-prefix=GFX900 %s

// CHECK:  .amdgpu_code_object_metadata
// CHECK:    Version: [ 1, 0 ]
// CHECK:    Kernels:
// CHECK:      - Name: test_kernel
// CHECK:        DebugProps:
// CHECK:          DebuggerABIVersion:                [ 1, 0 ]
// CHECK:          ReservedNumVGPRs:                  4
// CHECK:          ReservedFirstVGPR:                 11
// CHECK:          PrivateSegmentBufferSGPR:          0
// CHECK:          WavefrontPrivateSegmentOffsetSGPR: 11
.amdgpu_code_object_metadata
  Version: [ 1, 0 ]
  Printf: [ '1:1:4:%d\n', '2:1:8:%g\n' ]
  Kernels:
    - Name:            test_kernel
      DebugProps:
        DebuggerABIVersion:                [ 1, 0 ]
        ReservedNumVGPRs:                  4
        ReservedFirstVGPR:                 11
        PrivateSegmentBufferSGPR:          0
        WavefrontPrivateSegmentOffsetSGPR: 11
.end_amdgpu_code_object_metadata