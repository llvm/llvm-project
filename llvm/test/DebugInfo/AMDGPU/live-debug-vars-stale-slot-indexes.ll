; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx90a -O3 -stop-after=virtregrewriter,2 < %s | FileCheck %s

; LiveDebugVariables is built before the SGPR allocator and still holds
; SlotIndexes when the VGPR allocator spills %v1 several passes later. The two
; intervals recorded for %v1 around the spill resolve onto the same position, so
; only one DBG_VALUE belongs there. Without canonicalizeIndexes() each interval
; emits its own and the last CHECK-NEXT lands on the duplicate.

; CHECK-LABEL: name: partial_copy
; CHECK:      DBG_VALUE $vgpr0_vgpr1, $noreg, ![[V1:[0-9]+]], !DIExpression()
; CHECK-NEXT: SI_SPILL_AV64_SAVE
; CHECK-NEXT: DBG_VALUE %stack.0, 0, ![[V1]], !DIExpression()
; CHECK-NEXT: GLOBAL_STORE_DWORDX4

define amdgpu_kernel void @partial_copy(<4 x i32> %arg) #0 !dbg !5 {
  call void asm sideeffect "; use $0", "a"(i32 poison), !dbg !13
  %v0 = call <4 x i32> asm sideeffect "; def $0", "=v"(), !dbg !14
    #dbg_value(<4 x i32> %v0, !9, !DIExpression(), !14)
  %v1 = call <2 x i32> asm sideeffect "; def $0", "=v"(), !dbg !15
    #dbg_value(<2 x i32> %v1, !11, !DIExpression(), !15)
  %mai = tail call <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32 1, i32 2, <4 x i32> %arg, i32 0, i32 0, i32 0), !dbg !16
    #dbg_value(<4 x i32> %mai, !12, !DIExpression(), !16)
  store volatile <4 x i32> %v0, ptr addrspace(1) poison, align 16, !dbg !17
  store volatile <2 x i32> %v1, ptr addrspace(1) poison, align 8, !dbg !18
  store volatile <4 x i32> %mai, ptr addrspace(1) poison, align 16, !dbg !19
  ret void, !dbg !20
}

declare <4 x i32> @llvm.amdgcn.mfma.i32.4x4x4i8(i32, i32, <4 x i32>, i32, i32, i32)

; The VGPR budget is what forces %v1 to be spilled.
attributes #0 = { nounwind "amdgpu-num-vgpr"="5" "amdgpu-no-dispatch-id" "amdgpu-no-dispatch-ptr" "amdgpu-no-implicitarg-ptr" "amdgpu-no-lds-kernel-id" "amdgpu-no-queue-ptr" "amdgpu-no-workgroup-id-x" "amdgpu-no-cluster-id-x" "amdgpu-no-workgroup-id-y" "amdgpu-no-cluster-id-y" "amdgpu-no-workgroup-id-z" "amdgpu-no-cluster-id-z" "amdgpu-no-workitem-id-x" "amdgpu-no-workitem-id-y" "amdgpu-no-workitem-id-z" }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "llvm", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "live-debug-vars-stale-slot-indexes.c", directory: "/")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = distinct !DISubprogram(name: "partial_copy", scope: !1, file: !1, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!6 = !DISubroutineType(types: !2)
!7 = !DIBasicType(name: "ty128", size: 128, encoding: DW_ATE_unsigned)
!8 = !{!9, !11, !12}
!9 = !DILocalVariable(name: "v0", scope: !5, file: !1, line: 2, type: !7)
!10 = !DIBasicType(name: "ty64", size: 64, encoding: DW_ATE_unsigned)
!11 = !DILocalVariable(name: "v1", scope: !5, file: !1, line: 3, type: !10)
!12 = !DILocalVariable(name: "mai", scope: !5, file: !1, line: 4, type: !7)
!13 = !DILocation(line: 1, column: 1, scope: !5)
!14 = !DILocation(line: 2, column: 1, scope: !5)
!15 = !DILocation(line: 3, column: 1, scope: !5)
!16 = !DILocation(line: 4, column: 1, scope: !5)
!17 = !DILocation(line: 5, column: 1, scope: !5)
!18 = !DILocation(line: 6, column: 1, scope: !5)
!19 = !DILocation(line: 7, column: 1, scope: !5)
!20 = !DILocation(line: 8, column: 1, scope: !5)
