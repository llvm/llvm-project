--- |
  ; ModuleID = '/work/atimofee/sandbox/github/llvm-project/llvm/test/CodeGen/AMDGPU/NextUseAnalysis/dead-registers.mir'
  source_filename = "/work/atimofee/sandbox/github/llvm-project/llvm/test/CodeGen/AMDGPU/NextUseAnalysis/dead-registers.mir"
  target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
  target triple = "amdgcn-amd-amdhsa"
  
  define void @dead_registers() #0 {
  entry:
    unreachable
  }
  
  attributes #0 = { "target-cpu"="gfx90a" }
...
---
name:            dead_registers
alignment:       1
exposesReturnsTwice: false
legalized:       false
regBankSelected: false
selected:        false
failedISel:      false
tracksRegLiveness: true
hasWinCFI:       false
noPhis:          true
isSSA:           true
noVRegs:         false
hasFakeUses:     false
callsEHReturn:   false
callsUnwindInit: false
hasEHContTarget: false
hasEHScopes:     false
hasEHFunclets:   false
isOutlined:      false
debugInstrRef:   false
failsVerification: false
tracksDebugUserValues: false
registers:
  - { id: 0, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 1, class: vgpr_32, preferred-register: '', flags: [  ] }
  - { id: 2, class: vgpr_32, preferred-register: '', flags: [  ] }
liveins:         []
frameInfo:
  isFrameAddressTaken: false
  isReturnAddressTaken: false
  hasStackMap:     false
  hasPatchPoint:   false
  stackSize:       0
  offsetAdjustment: 0
  maxAlignment:    1
  adjustsStack:    false
  hasCalls:        false
  stackProtector:  ''
  functionContext: ''
  maxCallFrameSize: 4294967295
  cvBytesOfCalleeSavedRegisters: 0
  hasOpaqueSPAdjustment: false
  hasVAStart:      false
  hasMustTailInVarArgFunc: false
  hasTailCall:     false
  isCalleeSavedInfoValid: false
  localFrameSize:  0
fixedStack:      []
stack:           []
entry_values:    []
callSites:       []
debugValueSubstitutions: []
constants:       []
machineFunctionInfo:
  explicitKernArgSize: 0
  maxKernArgAlign: 1
  ldsSize:         0
  gdsSize:         0
  dynLDSAlign:     1
  isEntryFunction: false
  isChainFunction: false
  noSignedZerosFPMath: false
  memoryBound:     false
  waveLimiter:     false
  hasSpilledSGPRs: false
  hasSpilledVGPRs: false
  numWaveDispatchSGPRs: 0
  numWaveDispatchVGPRs: 0
  scratchRSrcReg:  '$private_rsrc_reg'
  frameOffsetReg:  '$fp_reg'
  stackPtrOffsetReg: '$sp_reg'
  bytesInStackArgArea: 0
  returnsVoid:     true
  argumentInfo:
    privateSegmentBuffer: { reg: '$sgpr0_sgpr1_sgpr2_sgpr3' }
    dispatchPtr:     { reg: '$sgpr4_sgpr5' }
    queuePtr:        { reg: '$sgpr6_sgpr7' }
    dispatchID:      { reg: '$sgpr10_sgpr11' }
    workGroupIDX:    { reg: '$sgpr12' }
    workGroupIDY:    { reg: '$sgpr13' }
    workGroupIDZ:    { reg: '$sgpr14' }
    LDSKernelId:     { reg: '$sgpr15' }
    implicitArgPtr:  { reg: '$sgpr8_sgpr9' }
    workItemIDX:     { reg: '$vgpr31', mask: 1023 }
    workItemIDY:     { reg: '$vgpr31', mask: 1047552 }
    workItemIDZ:     { reg: '$vgpr31', mask: 1072693248 }
  psInputAddr:     0
  psInputEnable:   0
  maxMemoryClusterDWords: 8
  mode:
    ieee:            true
    dx10-clamp:      true
    fp32-input-denormals: true
    fp32-output-denormals: true
    fp64-fp16-input-denormals: true
    fp64-fp16-output-denormals: true
  highBitsOf32BitAddress: 0
  occupancy:       8
  vgprForAGPRCopy: ''
  sgprForEXECCopy: ''
  longBranchReservedReg: ''
  hasInitWholeWave: false
  dynamicVGPRBlockSize: 0
  scratchReservedForDynamicVGPRs: 0
  isWholeWaveFunction: false
body:             |
  bb.0:
    %0:vgpr_32 = V_MOV_B32_e32 42, implicit $exec
    %1:vgpr_32 = V_MOV_B32_e32 100, implicit $exec
    %2:vgpr_32 = V_ADD_F32_e32 %1, %1, implicit $exec, implicit $mode
    S_ENDPGM 0
...
