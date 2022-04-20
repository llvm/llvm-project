; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -stop-after=si-pre-allocate-wwm-regs -o %t.mir %s
; RUN: llc -run-pass=none -verify-machineinstrs %t.mir -o - | FileCheck %s

; Test that SIMachineFunctionInfo can be round trip serialized through
; MIR.

@lds = addrspace(3) global [512 x float] undef, align 4

; CHECK-LABEL: {{^}}name: kernel
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 128
; CHECK-NEXT: maxKernArgAlign: 64
; CHECK-NEXT: ldsSize: 2048
; CHECK-NEXT: gdsSize: 0
; CHECK-NEXT: dynLDSAlign: 1
; CHECK-NEXT: isEntryFunction: true
; CHECK-NEXT: noSignedZerosFPMath: false
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: hasSpilledSGPRs: false
; CHECK-NEXT: hasSpilledVGPRs: false
; CHECK-NEXT: scratchRSrcReg:  '$sgpr96_sgpr97_sgpr98_sgpr99'
; CHECK-NEXT: frameOffsetReg:  '$fp_reg'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT: bytesInStackArgArea: 0
; CHECK-NEXT: returnsVoid: true
; CHECK-NEXT: argumentInfo:
; CHECK-NEXT: privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; CHECK-NEXT: kernargSegmentPtr: { reg: '$sgpr4_sgpr5' }
; CHECK-NEXT: workGroupIDX: { reg: '$sgpr6' }
; CHECK-NEXT: privateSegmentWaveByteOffset: { reg: '$sgpr7' }
; CHECK-NEXT: workItemIDX: { reg: '$vgpr0' }
; CHECK-NEXT: mode:
; CHECK-NEXT: ieee: true
; CHECK-NEXT: dx10-clamp: true
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
; CHECK-NEXT: highBitsOf32BitAddress: 0
; CHECK-NEXT: occupancy: 10
; CHECK-NEXT: body:
define amdgpu_kernel void @kernel(i32 %arg0, i64 %arg1, <16 x i32> %arg2) {
  %gep = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %arg0
  store float 0.0, float addrspace(3)* %gep, align 4
  ret void
}

@gds = addrspace(2) global [128 x i32] undef, align 4

; CHECK-LABEL: {{^}}name: ps_shader
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 0
; CHECK-NEXT: maxKernArgAlign: 4
; CHECK-NEXT: ldsSize: 0
; CHECK-NEXT: gdsSize: 0
; CHECK-NEXT: dynLDSAlign: 1
; CHECK-NEXT: isEntryFunction: true
; CHECK-NEXT: noSignedZerosFPMath: false
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: hasSpilledSGPRs: false
; CHECK-NEXT: hasSpilledVGPRs: false
; CHECK-NEXT: scratchRSrcReg:  '$sgpr96_sgpr97_sgpr98_sgpr99'
; CHECK-NEXT: frameOffsetReg:  '$fp_reg'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT: bytesInStackArgArea: 0
; CHECK-NEXT: returnsVoid: true
; CHECK-NEXT: argumentInfo:
; CHECK-NEXT: privateSegmentWaveByteOffset: { reg: '$sgpr3' }
; CHECK-NEXT: implicitBufferPtr: { reg: '$sgpr0_sgpr1' }
; CHECK-NEXT: mode:
; CHECK-NEXT: ieee: false
; CHECK-NEXT: dx10-clamp: true
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
; CHECK-NEXT: highBitsOf32BitAddress: 0
; CHECK-NEXT: occupancy: 10
; CHECK-NEXT: body:
define amdgpu_ps void @ps_shader(i32 %arg0, i32 inreg %arg1) {
  ret void
}

; CHECK-LABEL: {{^}}name: gds_size_shader
; CHECK: gdsSize: 4096
define amdgpu_ps void @gds_size_shader(i32 %arg0, i32 inreg %arg1) #5 {
  ret void
}

; CHECK-LABEL: {{^}}name: function
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 0
; CHECK-NEXT: maxKernArgAlign: 1
; CHECK-NEXT: ldsSize: 0
; CHECK-NEXT: gdsSize: 0
; CHECK-NEXT: dynLDSAlign: 1
; CHECK-NEXT: isEntryFunction: false
; CHECK-NEXT: noSignedZerosFPMath: false
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: hasSpilledSGPRs: false
; CHECK-NEXT: hasSpilledVGPRs: false
; CHECK-NEXT: scratchRSrcReg: '$sgpr0_sgpr1_sgpr2_sgpr3'
; CHECK-NEXT: frameOffsetReg: '$sgpr33'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT: bytesInStackArgArea: 0
; CHECK-NEXT: returnsVoid: true
; CHECK-NEXT: argumentInfo:
; CHECK-NEXT: privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; CHECK-NEXT: dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; CHECK-NEXT: queuePtr:        { reg: '$sgpr6_sgpr7' }
; CHECK-NEXT: dispatchID:      { reg: '$sgpr10_sgpr11' }
; CHECK-NEXT: workGroupIDX:    { reg: '$sgpr12' }
; CHECK-NEXT: workGroupIDY:    { reg: '$sgpr13' }
; CHECK-NEXT: workGroupIDZ:    { reg: '$sgpr14' }
; CHECK-NEXT: implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
; CHECK-NEXT: workItemIDX:     { reg: '$vgpr31', mask: 1023 }
; CHECK-NEXT: workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
; CHECK-NEXT: workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
; CHECK-NEXT: mode:
; CHECK-NEXT: ieee: true
; CHECK-NEXT: dx10-clamp: true
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
; CHECK-NEXT: highBitsOf32BitAddress: 0
; CHECK-NEXT: occupancy: 10
; CHECK-NEXT: body:
define void @function() {
  ret void
}

; CHECK-LABEL: {{^}}name: function_nsz
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 0
; CHECK-NEXT: maxKernArgAlign: 1
; CHECK-NEXT: ldsSize: 0
; CHECK-NEXT: gdsSize: 0
; CHECK-NEXT: dynLDSAlign: 1
; CHECK-NEXT: isEntryFunction: false
; CHECK-NEXT: noSignedZerosFPMath: true
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: hasSpilledSGPRs: false
; CHECK-NEXT: hasSpilledVGPRs: false
; CHECK-NEXT: scratchRSrcReg: '$sgpr0_sgpr1_sgpr2_sgpr3'
; CHECK-NEXT: frameOffsetReg: '$sgpr33'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT: bytesInStackArgArea: 0
; CHECK-NEXT: returnsVoid: true
; CHECK-NEXT: argumentInfo:
; CHECK-NEXT: privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
; CHECK-NEXT: dispatchPtr:     { reg: '$sgpr4_sgpr5' }
; CHECK-NEXT: queuePtr:        { reg: '$sgpr6_sgpr7' }
; CHECK-NEXT: dispatchID:      { reg: '$sgpr10_sgpr11' }
; CHECK-NEXT: workGroupIDX:    { reg: '$sgpr12' }
; CHECK-NEXT: workGroupIDY:    { reg: '$sgpr13' }
; CHECK-NEXT: workGroupIDZ:    { reg: '$sgpr14' }
; CHECK-NEXT: implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
; CHECK-NEXT: workItemIDX:     { reg: '$vgpr31', mask: 1023 }
; CHECK-NEXT: workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
; CHECK-NEXT: workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
; CHECK-NEXT: mode:
; CHECK-NEXT: ieee: true
; CHECK-NEXT: dx10-clamp: true
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
; CHECK-NEXT: highBitsOf32BitAddress: 0
; CHECK-NEXT: occupancy: 10
; CHECK-NEXT: body:
define void @function_nsz() #0 {
  ret void
}

; CHECK-LABEL: {{^}}name: function_dx10_clamp_off
; CHECK: mode:
; CHECK-NEXT: ieee: true
; CHECK-NEXT: dx10-clamp: false
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
define void @function_dx10_clamp_off() #1 {
  ret void
}

; CHECK-LABEL: {{^}}name: function_ieee_off
; CHECK: mode:
; CHECK-NEXT: ieee: false
; CHECK-NEXT: dx10-clamp: true
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
define void @function_ieee_off() #2 {
  ret void
}

; CHECK-LABEL: {{^}}name: function_ieee_off_dx10_clamp_off
; CHECK: mode:
; CHECK-NEXT: ieee: false
; CHECK-NEXT: dx10-clamp: false
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
define void @function_ieee_off_dx10_clamp_off() #3 {
  ret void
}

; CHECK-LABEL: {{^}}name: high_address_bits
; CHECK: machineFunctionInfo:
; CHECK: highBitsOf32BitAddress: 4294934528
define amdgpu_ps void @high_address_bits() #4 {
  ret void
}

; CHECK-LABEL: {{^}}name: wwm_reserved_regs
; CHECK: wwmReservedRegs:
; CHECK-NEXT: - '$vgpr2'
; CHECK-NEXT: - '$vgpr3'
define amdgpu_cs void @wwm_reserved_regs(i32 addrspace(1)* %ptr, <4 x i32> inreg %tmp14) {
  %ld0 = load volatile i32, i32 addrspace(1)* %ptr
  %ld1 = load volatile i32, i32 addrspace(1)* %ptr
  %inactive0 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %ld1, i32 0)
  %inactive1 = tail call i32 @llvm.amdgcn.set.inactive.i32(i32 %ld0, i32 0)
  store volatile i32 %inactive0, i32 addrspace(1)* %ptr
  store volatile i32 %inactive1, i32 addrspace(1)* %ptr
  ret void
}

declare i32 @llvm.amdgcn.set.inactive.i32(i32, i32) #6

attributes #0 = { "no-signed-zeros-fp-math" = "true" }
attributes #1 = { "amdgpu-dx10-clamp" = "false" }
attributes #2 = { "amdgpu-ieee" = "false" }
attributes #3 = { "amdgpu-dx10-clamp" = "false" "amdgpu-ieee" = "false" }
attributes #4 = { "amdgpu-32bit-address-high-bits"="0xffff8000" }
attributes #5 = { "amdgpu-gds-size"="4096" }
attributes #6 = { convergent nounwind readnone willreturn }
