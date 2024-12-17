; RUN: llc -mtriple=amdgcn-amd-amdhsa -verify-machineinstrs -amdgpu-s-branch-bits=4 -stop-after=branch-relaxation -verify-machineinstrs %s -o - | FileCheck %s

; Test that debug instructions do not change long branch reserved serialized through
; MIR.

; CHECK-LABEL: {{^}}name: uniform_long_forward_branch_debug
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
; CHECK-NEXT: maxMemoryClusterDWords: 8
; CHECK-NEXT: mode:
; CHECK-NEXT: ieee:            true
; CHECK-NEXT: dx10-clamp:      true
; CHECK-NEXT: fp32-input-denormals: true
; CHECK-NEXT: fp32-output-denormals: true
; CHECK-NEXT: fp64-fp16-input-denormals: true
; CHECK-NEXT: fp64-fp16-output-denormals: true
; CHECK-NEXT: BitsOf32BitAddress: 0
; CHECK-NEXT: occupancy:       8
; CHECK-NEXT: vgprForAGPRCopy: ''
; CHECK-NEXT: sgprForEXECCopy: '$sgpr100_sgpr101'
; CHECK-NEXT: longBranchReservedReg: '$sgpr2_sgpr3'
; CHECK-NEXT: hasInitWholeWave: false
; CHECK-NEXT: body:
  define amdgpu_kernel void @uniform_long_forward_branch_debug(ptr addrspace(1) %arg, i32 %arg1) #0 !dbg !5 {
  bb0:
    %uniform_long_forward_branch_debug.kernarg.segment = call nonnull align 16 dereferenceable(12) ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr(), !dbg !11
    %arg1.kernarg.offset = getelementptr inbounds i8, ptr addrspace(4) %uniform_long_forward_branch_debug.kernarg.segment, i64 8, !dbg !11, !amdgpu.uniform !7
    %arg1.load = load i32, ptr addrspace(4) %arg1.kernarg.offset, align 8, !dbg !11, !invariant.load !7
    %tmp = icmp eq i32 %arg1.load, 0, !dbg !11
    call void @llvm.dbg.value(metadata i1 %tmp, metadata !9, metadata !DIExpression()), !dbg !11
    br i1 %tmp, label %bb3, label %Flow, !dbg !12, !amdgpu.uniform !7

  Flow:                                             ; preds = %bb3, %bb0
    %0 = phi i1 [ false, %bb3 ], [ true, %bb0 ], !dbg !12
    br i1 %0, label %bb2, label %bb4, !dbg !12, !amdgpu.uniform !7

  bb2:                                              ; preds = %Flow
    store volatile i32 17, ptr addrspace(1) undef, align 4, !dbg !13
    br label %bb4, !dbg !14, !amdgpu.uniform !7

  bb3:                                              ; preds = %bb0
    call void asm sideeffect "v_nop_e64\0A  v_nop_e64\0A  v_nop_e64\0A  v_nop_e64", ""(), !dbg !15
    br label %Flow, !dbg !16, !amdgpu.uniform !7

  bb4:                                              ; preds = %bb2, %Flow
    %arg.kernarg.offset1 = bitcast ptr addrspace(4) %uniform_long_forward_branch_debug.kernarg.segment to ptr addrspace(4), !dbg !11, !amdgpu.uniform !7
    %arg.load = load ptr addrspace(1), ptr addrspace(4) %arg.kernarg.offset1, align 16, !dbg !11, !invariant.load !7
    store volatile i32 63, ptr addrspace(1) %arg.load, align 4, !dbg !17
    ret void, !dbg !18
  }

  ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
  declare void @llvm.dbg.value(metadata, metadata, metadata) #1

  ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
  declare align 4 ptr addrspace(4) @llvm.amdgcn.kernarg.segment.ptr() #1

  ; Function Attrs: convergent nocallback nofree nounwind willreturn
  declare { i1, i64 } @llvm.amdgcn.if.i64(i1) #2

  ; Function Attrs: convergent nocallback nofree nounwind willreturn
  declare { i1, i64 } @llvm.amdgcn.else.i64.i64(i64) #2

  ; Function Attrs: convergent nocallback nofree nounwind willreturn memory(none)
  declare i64 @llvm.amdgcn.if.break.i64(i1, i64) #3

  ; Function Attrs: convergent nocallback nofree nounwind willreturn
  declare i1 @llvm.amdgcn.loop.i64(i64) #2

  ; Function Attrs: convergent nocallback nofree nounwind willreturn
  declare void @llvm.amdgcn.end.cf.i64(i64) #2

  attributes #0 = { "amdgpu-no-completion-action" "amdgpu-no-default-queue" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-heap-ptr" "amdgpu-no-hostcall-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-multigrid-sync-arg" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" "uniform-work-group-size"="false" }
  attributes #1 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
  attributes #2 = { convergent nocallback nofree nounwind willreturn }
  attributes #3 = { convergent nocallback nofree nounwind willreturn memory(none) }

  !llvm.dbg.cu = !{!0}
  !llvm.debugify = !{!2, !3}
  !llvm.module.flags = !{!4}

  !0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "debugify", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
  !1 = !DIFile(filename: "temp.ll", directory: "/")
  !2 = !{i32 8}
  !3 = !{i32 1}
  !4 = !{i32 2, !"Debug Info Version", i32 3}
  !5 = distinct !DISubprogram(name: "uniform_long_forward_branch_debug", linkageName: "uniform_long_forward_branch_debug", scope: null, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
  !6 = !DISubroutineType(types: !7)
  !7 = !{}
  !8 = !{!9}
  !9 = !DILocalVariable(name: "1", scope: !5, file: !1, line: 1, type: !10)
  !10 = !DIBasicType(name: "ty8", size: 8, encoding: DW_ATE_unsigned)
  !11 = !DILocation(line: 1, column: 1, scope: !5)
  !12 = !DILocation(line: 2, column: 1, scope: !5)
  !13 = !DILocation(line: 3, column: 1, scope: !5)
  !14 = !DILocation(line: 4, column: 1, scope: !5)
  !15 = !DILocation(line: 5, column: 1, scope: !5)
  !16 = !DILocation(line: 6, column: 1, scope: !5)
  !17 = !DILocation(line: 7, column: 1, scope: !5)
  !18 = !DILocation(line: 8, column: 1, scope: !5)
