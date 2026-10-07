; Verifies that debug variable locations for both SGPR and VGPR register
; classes are preserved through the AMDGPU multi-stage register allocation
; pipeline in non-instruction-referencing mode.
;
; The AMDGPU backend allocates registers in three stages: SGPR → WWM → VGPR.
; Each stage has its own Greedy Allocator and VirtRegRewriter. Previously,
; VRM mappings from the SGPR stage were lost before the final emission because
; intervening passes (e.g. StackSlotColoring) did not preserve VRM. The fix
; adds resolveAssignedLocations() calls at intermediate VirtRegRewriter passes,
; capturing assigned physregs into LDV's UserValue data before VRM is cleared.
;
; AMDGPU does NOT use instruction-referencing mode by default — only x86_64
; does (see debuginfoShouldUseDebugInstrRef() in LiveDebugValues.cpp). This
; means ALL DBG_VALUE and DBG_VALUE_LIST instructions on AMDGPU are routed
; through LDV's UserValue path, making them directly susceptible to VRM loss
; across RA stages. The fix is therefore critical for all debug values on
; AMDGPU, not just edge cases.
;
; The -experimental-debug-variable-locations=false flag explicitly forces
; non-instr-ref mode, matching AMDGPU's actual default behavior. This routes
; ALL DBG_VALUE instructions through UserValue (not stashing).
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -O1 \
; RUN:   -experimental-debug-variable-locations=false \
; RUN:   -print-after=virtregrewriter -o /dev/null < %s 2>&1 | \
; RUN:   FileCheck %s

; --- Function 1: DBG_VALUE for separate SGPR and VGPR variables ---
; The NoVRegs flag distinguishes the final (VGPR) VirtRegRewriter dump.
; CHECK: Machine code for function test_sgpr_vgpr_dbg_value:{{.*}}NoVRegs
; CHECK: DBG_VALUE $sgpr{{[0-9]+}}, $noreg, !"uniform_val"
; CHECK: DBG_VALUE $vgpr{{[0-9]+}}, $noreg, !"divergent_val"

; --- Function 2: DBG_VALUE_LIST with mixed SGPR + VGPR operands ---
; CHECK: Machine code for function test_dbg_value_list_mixed:{{.*}}NoVRegs
; CHECK: DBG_VALUE_LIST !"combined", !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value), $sgpr{{[0-9]+}}, $vgpr{{[0-9]+}}

define amdgpu_kernel void @test_sgpr_vgpr_dbg_value(
    ptr addrspace(1) %out,
    i32 %uniform_arg
) #0 !dbg !6 {
entry:
  %tid = call i32 @llvm.amdgcn.workitem.id.x()

  %uniform_val = add i32 %uniform_arg, 42
  call void @llvm.dbg.value(metadata i32 %uniform_val, metadata !9, metadata !DIExpression()), !dbg !14

  %divergent_val = add i32 %tid, %uniform_val
  call void @llvm.dbg.value(metadata i32 %divergent_val, metadata !10, metadata !DIExpression()), !dbg !14

  %gep = getelementptr i32, ptr addrspace(1) %out, i32 %tid
  store i32 %divergent_val, ptr addrspace(1) %gep, align 4
  ret void
}

; Non-kernel function: inreg → SGPR, normal arg → VGPR.
; DIArgList produces a DBG_VALUE_LIST referencing both register classes.
define void @test_dbg_value_list_mixed(
    i32 inreg %sgpr_arg,
    i32 %vgpr_arg,
    ptr addrspace(1) %out
) #0 !dbg !20 {
entry:
  call void @llvm.dbg.value(metadata !DIArgList(i32 %sgpr_arg, i32 %vgpr_arg), metadata !22, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_plus, DW_OP_stack_value)), !dbg !24
  %sum = add i32 %sgpr_arg, %vgpr_arg
  store i32 %sum, ptr addrspace(1) %out
  ret void
}

declare i32 @llvm.amdgcn.workitem.id.x() #1
declare void @llvm.dbg.value(metadata, metadata, metadata) #1

attributes #0 = { nounwind "amdgpu-flat-work-group-size"="64,256" }
attributes #1 = { nounwind readnone speculatable willreturn }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cl", directory: "/tmp")
!2 = !{}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}

; Metadata for test_sgpr_vgpr_dbg_value
!6 = distinct !DISubprogram(name: "test_sgpr_vgpr_dbg_value", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9, !10}
!9 = !DILocalVariable(name: "uniform_val", scope: !6, file: !1, line: 2, type: !12)
!10 = !DILocalVariable(name: "divergent_val", scope: !6, file: !1, line: 3, type: !12)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!14 = !DILocation(line: 2, column: 1, scope: !6)

; Metadata for test_dbg_value_list_mixed
!20 = distinct !DISubprogram(name: "test_dbg_value_list_mixed", scope: !1, file: !1, line: 10, type: !7, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!21 = !{!22}
!22 = !DILocalVariable(name: "combined", scope: !20, file: !1, line: 11, type: !12)
!24 = !DILocation(line: 11, column: 1, scope: !20)
