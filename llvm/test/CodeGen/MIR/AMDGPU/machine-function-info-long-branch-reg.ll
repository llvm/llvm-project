; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -amdgpu-s-branch-bits=4 -stop-after=branch-relaxation -verify-machineinstrs %s -o - | FileCheck %s

; Test that long branch reserved register is serialized through
; MIR.

; CHECK-LABEL: {{^}}name: uniform_long_forward_branch
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 12
; CHECK-NEXT: maxKernArgAlign: 8
; CHECK-NEXT: ldsSize: 0
; CHECK-NEXT: gdsSize: 0
; CHECK-NEXT: dynLDSAlign: 1
; CHECK-NEXT: isEntryFunction: true
; CHECK-NEXT: isChainFunction: false
; CHECK-NEXT: noSignedZerosFPMath: false
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: hasSpilledSGPRs: false
; CHECK-NEXT: hasSpilledVGPRs: false
; CHECK-NEXT: scratchRSrcReg:  '$sgpr96_sgpr97_sgpr98_sgpr99'
; CHECK-NEXT: frameOffsetReg:  '$fp_reg'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT: bytesInStackArgArea: 0
; CHECK-NEXT: returnsVoid:     true
; CHECK-NEXT: argumentInfo:
; CHECK-NEXT: privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; CHECK-NEXT: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; CHECK-NEXT: workGroupIDX:    { reg: '$sgpr6' }
; CHECK-NEXT: privateSegmentWaveByteOffset: { reg: '$sgpr7' }
; CHECK-NEXT: workItemIDX:     { reg: '$vgpr0' }
; CHECK-NEXT: psInputAddr:     0
; CHECK-NEXT: psInputEnable:   0
; CHECK-NEXT: maxMemoryClusterDWords:   8
; CHECK-NEXT: mode:
; CHECK-NEXT: ieee:            true
; CHECK-NEXT: dx10-clamp:      true
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
; CHECK-NEXT: BitsOf32BitAddress: 0
; CHECK-NEXT: occupancy:       10
; CHECK-NEXT: vgprForAGPRCopy: ''
; CHECK-NEXT: sgprForEXECCopy: '$sgpr100_sgpr101'
; CHECK-NEXT: longBranchReservedReg: '$sgpr2_sgpr3'
; CHECK-NEXT: hasInitWholeWave: false
; CHECK-NEXT: dynamicVGPRBlockSize: 0
; CHECK-NEXT: scratchReservedForDynamicVGPRs: 0
; CHECK-NEXT: isWholeWaveFunction: false
; CHECK-NEXT: body:
define amdgpu_kernel void @uniform_long_forward_branch(ptr addrspace(1) %arg, i32 %arg1) #0 {
bb0:
  %tmp = icmp ne i32 %arg1, 0
  br i1 %tmp, label %bb2, label %bb3

bb2:
  store volatile i32 17, ptr addrspace(1) poison
  br label %bb4

bb3:
  ; 32 byte asm
  call void asm sideeffect
  "v_nop_e64
  v_nop_e64
  v_nop_e64
  v_nop_e64", ""() #0
  br label %bb4

bb4:
  store volatile i32 63, ptr addrspace(1) %arg
  ret void
}

attributes #0 = { nounwind "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-flat-scratch-init" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }
attributes #1 = { nounwind readnone }
