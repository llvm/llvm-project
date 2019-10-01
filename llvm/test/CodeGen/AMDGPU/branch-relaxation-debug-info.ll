; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 -amdgpu-s-branch-bits=4 -verify-machineinstrs < %s | FileCheck -check-prefix=GCN %s

; Make sure there is no assertion due to dbg_value instructions
; present in the block used for the branch expansion.

declare void @llvm.dbg.value(metadata, metadata, metadata) #0

define amdgpu_kernel void @long_branch_dbg_value(float addrspace(1)* nocapture %arg, float %arg1) #1 !dbg !5 {
; GCN-LABEL: long_branch_dbg_value:
; GCN:  BB0_4: ; %bb
; GCN-NEXT:    ;DEBUG_VALUE: test_debug_value:globalptr_arg <- [DW_OP_plus_uconst 12, DW_OP_stack_value]
; GCN-NEXT:    .loc 1 0 42 is_stmt 0 ; /tmp/test_debug_value.cl:0:42
; GCN-NEXT:    s_getpc_b64 s{{\[}}[[PC_LO:[0-9]+]]:[[PC_HI:[0-9]+]]{{\]}}
; GCN-NEXT:    s_add_u32 s[[PC_LO]], s[[PC_LO]], BB0_3-(BB0_4+4)
; GCN-NEXT:    s_addc_u32 s[[PC_HI]], s[[PC_HI]], 0
; GCN-NEXT:    s_setpc_b64
bb:
  %tmp = fmul float %arg1, %arg1
  %tmp2 = getelementptr inbounds float, float addrspace(1)* %arg, i64 3
  call void @llvm.dbg.value(metadata float addrspace(1)* %tmp2, metadata !11, metadata !DIExpression()) #2, !dbg !12
  store float %tmp, float addrspace(1)* %tmp2, align 4, !dbg !12
  %tmp3 = fcmp olt float %tmp, 0x3810000000000000
  br i1 %tmp3, label %bb8, label %bb4

bb4:                                              ; preds = %bb
  %tmp5 = load volatile float, float addrspace(1)* undef
  %tmp6 = fcmp oeq float %tmp5, 0x7FF0000000000000
  br i1 %tmp6, label %bb7, label %bb8

bb7:                                              ; preds = %bb4
  br label %bb8

bb8:                                              ; preds = %bb7, %bb4, %bb
  ret void
}

attributes #0 = { nounwind readnone speculatable }
attributes #1 = { nounwind writeonly }
attributes #2 = { nounwind }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 3.8.0 (trunk 244715) (llvm/trunk 244718)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "/tmp/test_debug_value.cl", directory: "/Users/matt/src/llvm/build_debug")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = distinct !DISubprogram(name: "test_debug_value", scope: !1, file: !1, line: 1, type: !6, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !10)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !9, size: 64, align: 32)
!9 = !DIBasicType(name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!10 = !{!11}
!11 = !DILocalVariable(name: "globalptr_arg", arg: 1, scope: !5, file: !1, line: 1, type: !8)
!12 = !DILocation(line: 1, column: 42, scope: !5)
