; RUN: llc -mtriple=amdgcn--amdhsa -filetype=obj -o - < %s | llvm-readobj -amdgpu-runtime-metadata | FileCheck %s
; check llc does not crash for invalid opencl version metadata

; CHECK: { amd.MDVersion: [ 2, 1 ], amd.IsaInfo: { amd.IsaInfoWavefrontSize: 64, amd.IsaInfoLocalMemorySize: 65536, amd.IsaInfoEUsPerCU: 4, amd.IsaInfoMaxWavesPerEU: 10, amd.IsaInfoMaxFlatWorkGroupSize: 2048, amd.IsaInfoSGPRAllocGranule: 8, amd.IsaInfoTotalNumSGPRs: 512, amd.IsaInfoAddressableNumSGPRs: 104, amd.IsaInfoVGPRAllocGranule: 4, amd.IsaInfoTotalNumVGPRs: 256, amd.IsaInfoAddressableNumVGPRs: 256 } }

!opencl.ocl.version = !{!0}
!0 = !{i32 1}
