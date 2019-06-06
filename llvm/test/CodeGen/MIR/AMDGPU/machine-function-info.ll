; RUN: llc -mtriple=amdgcn-mesa-mesa3d -mcpu=tahiti -stop-after expand-isel-pseudos -o %t.mir %s
; RUN: llc -run-pass=none -verify-machineinstrs %t.mir -o - | FileCheck %s

; Test that SIMachineFunctionInfo can be round trip serialized through
; MIR.

@lds = addrspace(3) global [512 x float] undef, align 4

; CHECK-LABEL: {{^}}name: kernel
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 128
; CHECK-NEXT: maxKernArgAlign: 64
; CHECK-NEXT: ldsSize: 2048
; CHECK-NEXT: isEntryFunction: true
; CHECK-NEXT: noSignedZerosFPMath: false
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: scratchRSrcReg:  '$sgpr96_sgpr97_sgpr98_sgpr99'
; CHECK-NEXT: scratchWaveOffsetReg: '$sgpr101'
; CHECK-NEXT: frameOffsetReg:  '$sgpr101'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr101'
; CHECK-NEXT: body:
define amdgpu_kernel void @kernel(i32 %arg0, i64 %arg1, <16 x i32> %arg2) {
  %gep = getelementptr inbounds [512 x float], [512 x float] addrspace(3)* @lds, i32 0, i32 %arg0
  store float 0.0, float addrspace(3)* %gep, align 4
  ret void
}

; CHECK-LABEL: {{^}}name: ps_shader
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 0
; CHECK-NEXT: maxKernArgAlign: 0
; CHECK-NEXT: ldsSize: 0
; CHECK-NEXT: isEntryFunction: true
; CHECK-NEXT: noSignedZerosFPMath: false
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: scratchRSrcReg:  '$sgpr96_sgpr97_sgpr98_sgpr99'
; CHECK-NEXT: scratchWaveOffsetReg: '$sgpr101'
; CHECK-NEXT: frameOffsetReg:  '$sgpr101'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr101'
; CHECK-NEXT: body:
define amdgpu_ps void @ps_shader(i32 %arg0, i32 inreg %arg1) {
  ret void
}

; CHECK-LABEL: {{^}}name: function
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 0
; CHECK-NEXT: maxKernArgAlign: 0
; CHECK-NEXT: ldsSize: 0
; CHECK-NEXT: isEntryFunction: false
; CHECK-NEXT: noSignedZerosFPMath: false
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: scratchRSrcReg: '$sgpr0_sgpr1_sgpr2_sgpr3'
; CHECK-NEXT: scratchWaveOffsetReg: '$sgpr4'
; CHECK-NEXT: frameOffsetReg: '$sgpr5'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT: body:
define void @function() {
  ret void
}

; CHECK-LABEL: {{^}}name: function_nsz
; CHECK: machineFunctionInfo:
; CHECK-NEXT: explicitKernArgSize: 0
; CHECK-NEXT: maxKernArgAlign: 0
; CHECK-NEXT: ldsSize: 0
; CHECK-NEXT: isEntryFunction: false
; CHECK-NEXT: noSignedZerosFPMath: true
; CHECK-NEXT: memoryBound: false
; CHECK-NEXT: waveLimiter: false
; CHECK-NEXT: scratchRSrcReg: '$sgpr0_sgpr1_sgpr2_sgpr3'
; CHECK-NEXT: scratchWaveOffsetReg: '$sgpr4'
; CHECK-NEXT: frameOffsetReg: '$sgpr5'
; CHECK-NEXT: stackPtrOffsetReg: '$sgpr32'
; CHECK-NEXT: body:
define void @function_nsz() #0 {
  ret void
}

attributes #0 = { "no-signed-zeros-fp-math" = "true" }
