; Verifies that DBG_PHI instructions referencing SGPR virtual registers are
; correctly resolved to physical registers through the AMDGPU multi-stage
; register allocation pipeline in instruction-referencing mode.
;
; AMDGPU does NOT use instruction-referencing mode by default — only x86_64
; does (see debuginfoShouldUseDebugInstrRef() in LiveDebugValues.cpp). This
; test uses -experimental-debug-variable-locations=true to explicitly enable
; it, exercising the DBG_PHI / DBG_INSTR_REF code paths that would otherwise
; be dormant on AMDGPU. These paths become relevant if instr-ref mode is
; enabled for AMDGPU in the future.
;
; In instr-ref mode, PHI node debug info is tracked via DBG_PHI + DBG_INSTR_REF
; pairs. PHIElimination records debug PHI positions (mapping instruction numbers
; to their destination vregs). LDV stashes these in PHIValToPos and emits
; DBG_PHI instructions with resolved physregs at final emission.
;
; Without resolveAssignedLocations(), the SGPR vreg-to-physreg mapping from the
; first RA stage is lost when VRM is re-initialized for subsequent stages,
; causing DBG_PHI to emit $noreg instead of the correct $sgprN.
;
; The test also includes a DIArgList-based debug value that, in instr-ref mode,
; is translated by ISel into DBG_PHI + DBG_INSTR_REF pairs. This exercises the
; same PHIValToPos resolution path for multi-operand debug expressions.
;
; RUN: llc -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1030 -O1 \
; RUN:   -experimental-debug-variable-locations=true \
; RUN:   -print-after=virtregrewriter -o /dev/null < %s 2>&1 | \
; RUN:   FileCheck %s

; --- Function 1: SGPR PHI resolved via DBG_PHI ---
; The PHI result (uniform add across a uniform branch) lives in an SGPR vreg.
; After the fix, resolveAssignedLocations() pre-resolves it before VRM reset.
; CHECK: Machine code for function test_sgpr_phi:{{.*}}NoVRegs
; CHECK: DBG_PHI $sgpr{{[0-9]+}}, 1
; CHECK: DBG_INSTR_REF !"phi_result", !DIExpression(DW_OP_LLVM_arg, 0), dbg-instr-ref(1, 0)

; --- Function 2: DIArgList → DBG_PHI pair (SGPR + VGPR) ---
; In instr-ref mode, ISel lowers DIArgList into two DBG_PHI entries and a
; DBG_INSTR_REF that references both. The SGPR and VGPR entries must both
; carry physical registers, not $noreg.
; CHECK: Machine code for function test_diarglist_instr_ref:{{.*}}NoVRegs
; CHECK-DAG: DBG_PHI $sgpr{{[0-9]+}},
; CHECK-DAG: DBG_PHI $vgpr{{[0-9]+}},
; CHECK: DBG_INSTR_REF !"combined"

; Uniform branch with inline-asm side effects prevents if-conversion,
; preserving the SGPR PHI across register allocation.
define void @test_sgpr_phi(
    i32 inreg %cond,
    i32 inreg %a,
    i32 inreg %b,
    ptr addrspace(1) %out
) #0 !dbg !6 {
entry:
  %cmp = icmp sgt i32 %cond, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:
  call void asm sideeffect "", "~{s0}"()
  %va = add i32 %a, 10
  br label %if.end

if.else:
  call void asm sideeffect "", "~{s0}"()
  %vb = add i32 %b, 20
  br label %if.end

if.end:
  %result = phi i32 [%va, %if.then], [%vb, %if.else]
  call void @llvm.dbg.value(metadata i32 %result, metadata !9, metadata !DIExpression()), !dbg !11
  store i32 %result, ptr addrspace(1) %out
  ret void
}

; Non-kernel function: inreg → SGPR, normal arg → VGPR.
; In instr-ref mode, DIArgList is lowered to DBG_PHI + DBG_INSTR_REF.
define void @test_diarglist_instr_ref(
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

; Metadata for test_sgpr_phi
!6 = distinct !DISubprogram(name: "test_sgpr_phi", scope: !1, file: !1, line: 1, type: !7, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !8)
!7 = !DISubroutineType(types: !2)
!8 = !{!9}
!9 = !DILocalVariable(name: "phi_result", scope: !6, file: !1, line: 2, type: !12)
!11 = !DILocation(line: 2, column: 1, scope: !6)
!12 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

; Metadata for test_diarglist_instr_ref
!20 = distinct !DISubprogram(name: "test_diarglist_instr_ref", scope: !1, file: !1, line: 10, type: !7, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0, retainedNodes: !21)
!21 = !{!22}
!22 = !DILocalVariable(name: "combined", scope: !20, file: !1, line: 11, type: !12)
!24 = !DILocation(line: 11, column: 1, scope: !20)
