; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s | FileCheck %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s | FileCheck %s

; Check that debug values referencing eliminated frame indices don't crash.
; The AMDGPU backend can eliminate spill slots during frame finalization
; (e.g., SGPR spills to VGPR lanes). Debug values referencing these eliminated
; frame indices need to be cleaned up to avoid assertions in PrologEpilogInserter.

%struct.Buffer = type { [8 x i64] }

; CHECK-LABEL: test_dbg_value_dead_frame_idx:
; CHECK: ;DEBUG_VALUE: test_dbg_value_dead_frame_idx:slot <- [DW_OP_LLVM_arg 0, DW_OP_LLVM_arg 1, DW_OP_constu 64, DW_OP_mul, DW_OP_plus, DW_OP_stack_value] {{.*}}
; CHECK: s_endpgm
define amdgpu_kernel void @test_dbg_value_dead_frame_idx(ptr addrspace(1) %out, i64 %idx) !dbg !10 {
entry:
  #dbg_value(!DIArgList(ptr addrspace(1) %out, i64 %idx), !15, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 64, DW_OP_mul, DW_OP_plus, DW_OP_stack_value), !17)
  %ptr = getelementptr %struct.Buffer, ptr addrspace(1) %out, i64 %idx
  store i64 0, ptr addrspace(1) %ptr, align 8
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, dwarfAddressSpace: 1)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Buffer", file: !1, line: 1, size: 512, flags: DIFlagTypePassByValue, elements: !2)
!10 = distinct !DISubprogram(name: "test_dbg_value_dead_frame_idx", scope: !1, file: !1, line: 10, type: !11, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null, !6}
!15 = !DILocalVariable(name: "slot", scope: !10, file: !1, line: 11, type: !6)
!17 = !DILocation(line: 11, column: 1, scope: !10)
