; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx1100 < %s
; RUN: llc -O0 -mtriple=amdgcn-amd-amdhsa -mcpu=gfx900 < %s

; Check that debug values referencing eliminated frame indices don't crash.
; The AMDGPU backend can eliminate spill slots during frame finalization
; (e.g., SGPR spills to VGPR lanes). Debug values referencing these eliminated
; frame indices need to be cleaned up to avoid assertions in PrologEpilogInserter.

target datalayout = "e-m:e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%struct.Buffer = type { [8 x i64] }

; Test case with DIArgList and DW_OP_xderef that triggered the original crash.
; The dbg_value references a computed address that may involve frame indices
; from eliminated spill slots.
define ptr @test_dbg_value_dead_frame_idx() !dbg !10 {
entry:
  %idx = zext i32 0 to i64
  br label %body

body:
  #dbg_value(!DIArgList(ptr null, i64 %idx), !15, !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_constu, 64, DW_OP_mul, DW_OP_plus, DW_OP_constu, 1, DW_OP_swap, DW_OP_xderef, DW_OP_stack_value), !17)
  %ptr = getelementptr %struct.Buffer, ptr null, i64 %idx
  ret ptr %ptr
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 64, dwarfAddressSpace: 1)
!7 = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "Buffer", file: !1, line: 1, size: 512, flags: DIFlagTypePassByValue, elements: !2)
!10 = distinct !DISubprogram(name: "test_dbg_value_dead_frame_idx", scope: !1, file: !1, line: 10, type: !11, scopeLine: 10, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{!6}
!15 = !DILocalVariable(name: "slot", scope: !10, file: !1, line: 11, type: !6)
!17 = !DILocation(line: 11, column: 1, scope: !10)
