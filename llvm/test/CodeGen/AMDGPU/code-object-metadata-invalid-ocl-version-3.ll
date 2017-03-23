; RUN: llc -mtriple=amdgcn-amd-amdhsa -filetype=obj -o - < %s | llvm-readobj -amdgpu-code-object-metadata | FileCheck %s

; Make sure llc does not crash for invalid opencl version metadata.

; CHECK: ---
; CHECK: Version: [ 1, 0 ]
; CHECK: Isa:
; CHECK:   WavefrontSize:        64
; CHECK:   LocalMemorySize:      65536
; CHECK:   EUsPerCU:             4
; CHECK:   MaxWavesPerEU:        10
; CHECK:   MaxFlatWorkGroupSize: 2048
; CHECK:   SGPRAllocGranule:     8
; CHECK:   TotalNumSGPRs:        512
; CHECK:   AddressableNumSGPRs:  104
; CHECK:   VGPRAllocGranule:     4
; CHECK:   TotalNumVGPRs:        256
; CHECK:   AddressableNumVGPRs:  256
; CHECK: ...

!opencl.ocl.version = !{!0}
!0 = !{i32 1}
