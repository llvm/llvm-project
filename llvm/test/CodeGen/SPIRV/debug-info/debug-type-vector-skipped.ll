; RUN: llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Verify that DICompositeType vector nodes that cannot be lowered to
; DebugTypeVector are silently skipped. Four conditions trigger the skip:
;   1. No base type (getBaseType() returns null).
;   2. Non-DIBasicType base type (struct type as element type).
;   3. More than one DISubrange element (multi-dimensional array shape).
;   4. Non-constant subrange count (DILocalVariable used as component count).
;
; None of the four vectors below should produce a DebugTypeVector instruction.

; CHECK-NOT: DebugTypeVector

define spir_func void @test() !dbg !6 {
entry:
  %v1 = alloca <4 x float>, align 16
  %v2 = alloca <4 x float>, align 16
  %v3 = alloca <4 x i32>, align 16
  %v4 = alloca <4 x i64>, align 32
    #dbg_declare(ptr %v1, !10, !DIExpression(), !30)
    #dbg_declare(ptr %v2, !12, !DIExpression(), !30)
    #dbg_declare(ptr %v3, !15, !DIExpression(), !30)
    #dbg_declare(ptr %v4, !20, !DIExpression(), !30)
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_HLSL, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !DIFile(filename: "skip.hlsl", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !5)
!5 = !{null}
!6 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !9)
!9 = !{}

; Case 1: vector with no base type -- skipped at dyn_cast_or_null<DIBasicType>.
!10 = !DILocalVariable(name: "v1", scope: !6, file: !1, line: 2, type: !11)
!11 = !DICompositeType(tag: DW_TAG_array_type, size: 128, flags: DIFlagVector, elements: !29)

; Case 2: vector with non-DIBasicType base (struct) -- dyn_cast_or_null<DIBasicType>
; returns null, so skipped at the same check as case 1.
!12 = !DILocalVariable(name: "v2", scope: !6, file: !1, line: 3, type: !13)
!13 = !DICompositeType(tag: DW_TAG_array_type, baseType: !14, size: 128, flags: DIFlagVector, elements: !29)
!14 = !DICompositeType(tag: DW_TAG_structure_type, name: "S", file: !1, size: 32)

; Case 3: vector with multiple subranges -- skipped at Elements.size() != 1.
!15 = !DILocalVariable(name: "v3", scope: !6, file: !1, line: 4, type: !16)
!16 = !DICompositeType(tag: DW_TAG_array_type, baseType: !17, size: 128, flags: DIFlagVector, elements: !18)
!17 = !DIBasicType(name: "uint", size: 32, encoding: DW_ATE_unsigned)
!18 = !{!DISubrange(count: 4), !DISubrange(count: 4)}

; Case 4: vector with a non-constant subrange count (DILocalVariable) --
; skipped at dyn_cast_if_present<ConstantInt *>.
!20 = !DILocalVariable(name: "v4", scope: !6, file: !1, line: 5, type: !21)
!21 = !DICompositeType(tag: DW_TAG_array_type, baseType: !22, size: 256, flags: DIFlagVector, elements: !27)
!22 = !DIBasicType(name: "int64", size: 64, encoding: DW_ATE_signed)
!23 = !DILocalVariable(name: "n", scope: !6, file: !1, line: 1, type: !24)
!24 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!27 = !{!DISubrange(count: !23)}

!29 = !{!DISubrange(count: 4)}
!30 = !DILocation(line: 2, column: 1, scope: !6)
