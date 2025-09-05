; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -amdgpu-spill-sgpr-to-vgpr=0 -stop-after prologepilog -verify-machineinstrs %s -o - | FileCheck -check-prefix=AFTER-PEI %s

; Test that the ScavengeFI is serialized in the SIMachineFunctionInfo.

; AFTER-PEI-LABEL: {{^}}name: scavenge_fi
; AFTER-PEI: machineFunctionInfo:
; AFTER-PEI-NEXT: explicitKernArgSize: 12
; AFTER-PEI-NEXT: maxKernArgAlign: 8
; AFTER-PEI-NEXT: ldsSize:         0
; AFTER-PEI-NEXT: gdsSize:         0
; AFTER-PEI-NEXT: dynLDSAlign:     1
; AFTER-PEI-NEXT: isEntryFunction: true
; AFTER-PEI-NEXT: isChainFunction: false
; AFTER-PEI-NEXT: noSignedZerosFPMath: false
; AFTER-PEI-NEXT: memoryBound:     false
; AFTER-PEI-NEXT: waveLimiter:     false
; AFTER-PEI-NEXT: hasSpilledSGPRs: true
; AFTER-PEI-NEXT: hasSpilledVGPRs: false
; AFTER-PEI-NEXT: numWaveDispatchSGPRs: 0
; AFTER-PEI-NEXT: numWaveDispatchVGPRs: 0
; AFTER-PEI-NEXT: scratchRSrcReg:  '$sgpr68_sgpr69_sgpr70_sgpr71'
; AFTER-PEI-NEXT: frameOffsetReg:  '$fp_reg'
; AFTER-PEI-NEXT: stackPtrOffsetReg: '$sgpr32'
; AFTER-PEI-NEXT: bytesInStackArgArea: 0
; AFTER-PEI-NEXT: returnsVoid: true
; AFTER-PEI-NEXT: argumentInfo:
; AFTER-PEI-NEXT:   privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; AFTER-PEI-NEXT:   kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; AFTER-PEI-NEXT:   workGroupIDX:    { reg: '$sgpr6' }
; AFTER-PEI-NEXT:   privateSegmentWaveByteOffset: { reg: '$sgpr7' }
; AFTER-PEI-NEXT:   workItemIDX:     { reg: '$vgpr0' }
; AFTER-PEI-NEXT: psInputAddr:     0
; AFTER-PEI-NEXT: psInputEnable:   0
; AFTER-PEI-NEXT: maxMemoryClusterDWords:   8
; AFTER-PEI-NEXT: mode:
; AFTER-PEI-NEXT:   ieee:            true
; AFTER-PEI-NEXT:   dx10-clamp:      true
; AFTER-PEI-NEXT:   fp32-input-denormals: true
; AFTER-PEI-NEXT:   fp32-output-denormals: true
; AFTER-PEI-NEXT:   fp64-fp16-input-denormals: true
; AFTER-PEI-NEXT:   fp64-fp16-output-denormals: true
; AFTER-PEI-NEXT: highBitsOf32BitAddress: 0
; AFTER-PEI-NEXT: occupancy: 5
; AFTER-PEI-NEXT: scavengeFI: '%stack.3'
; AFTER-PEI-NEXT: vgprForAGPRCopy: ''
; AFTER-PEI-NEXT: sgprForEXECCopy: ''
; AFTER-PEI-NEXT: longBranchReservedReg: ''
; AFTER-PEI-NEXT: hasInitWholeWave: false
; AFTER-PEI-NEXT: dynamicVGPRBlockSize: 0
; AFTER-PEI-NEXT: scratchReservedForDynamicVGPRs: 0
; AFTER-PEI-NEXT: isWholeWaveFunction: false
; AFTER-PEI-NEXT: body:
define amdgpu_kernel void @scavenge_fi(ptr addrspace(1) %out, i32 %in) #0 {
  %wide.sgpr0 = call <32 x i32>  asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr1 = call <32 x i32>  asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr2 = call <32 x i32>  asm sideeffect "; def $0", "=s" () #0
  %wide.sgpr3 = call <32 x i32>  asm sideeffect "; def $0", "=s" () #0

  call void asm sideeffect "; use $0", "s"(<32 x i32> %wide.sgpr0) #0
  call void asm sideeffect "; use $0", "s"(<32 x i32> %wide.sgpr1) #0
  call void asm sideeffect "; use $0", "s"(<32 x i32> %wide.sgpr2) #0
  call void asm sideeffect "; use $0", "s"(<32 x i32> %wide.sgpr3) #0
  ret void
}

attributes #0 = { nounwind "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
