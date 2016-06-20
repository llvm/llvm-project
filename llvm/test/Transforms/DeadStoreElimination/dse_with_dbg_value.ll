; RUN: opt -basicaa -dse -S < %s | FileCheck %s
; RUN: opt -strip-debug -basicaa -dse -S < %s | FileCheck %s

; Test that stores are removed both with and without debug info.

; CHECK-NOT:  store i32 4, i32* @g_31, align 1
; CHECK-NOT:  %_tmp17.i.i = load i16, i16* %_tmp16.i.i, align 1
; CHECK-NOT:  store i16 %_tmp17.i.i, i16* @g_118, align 1
; CHECK:  store i32 0, i32* @g_31, align 1

@g_31 = global i32 0
@g_30 = global i16* null
@g_118 = global i16 0

define i16 @S0() !dbg !17 {
bb1:
  store i32 4, i32* @g_31, align 1, !dbg !20
  %_tmp16.i.i = load volatile i16*, i16** @g_30, align 1, !dbg !28
  %_tmp17.i.i = load i16, i16* %_tmp16.i.i, align 1, !dbg !28
  store i16 %_tmp17.i.i, i16* @g_118, align 1, !dbg !20
  store i32 0, i32* @g_31, align 1, !dbg !31
  tail call void @llvm.dbg.value(metadata i32 0, i64 0, metadata !40, metadata !41), !dbg !42
  store i16 0, i16* @g_118, align 1, !dbg !43
  br label %bb2.i, !dbg !44

bb2.i:
  br label %bb2.i, !dbg !44
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, i64, metadata, metadata) #0

attributes #0 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, producer: "FlexASIC FlexC Compiler v6.38 for FADER (LLVM)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, retainedTypes: !2, globals: !3)
!1 = !DIFile(filename: "csmith23219270180033.c", directory: "/local/repo/uabsson/llvm")
!2 = !{}
!3 = !{!4, !9, !13}
!4 = !DIGlobalVariable(name: "g_31", scope: null, file: !1, line: 3, type: !5, isLocal: false, isDefinition: true, variable: i32* @g_31)
!5 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !6, line: 104, baseType: !7)
!6 = !DIFile(filename: "/local/repo/uabsson/llvm/sdk-bin/cosy/fader2_sdk/compiler/fader2_arch/fader2_2/include/stdint.h", directory: "/local/repo/uabsson/llvm")
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "__i32_t", file: !1, baseType: !8)
!8 = !DIBasicType(name: "signed long", size: 32, align: 16, encoding: DW_ATE_signed)
!9 = !DIGlobalVariable(name: "g_30", scope: null, file: !1, line: 4, type: !10, isLocal: false, isDefinition: true, variable: i16** @g_30)
!10 = !DIDerivedType(tag: DW_TAG_volatile_type, baseType: !11)
!11 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !12, size: 16, align: 16)
!12 = !DIBasicType(name: "int", size: 16, align: 16, encoding: DW_ATE_signed)
!13 = !DIGlobalVariable(name: "g_118", scope: null, file: !1, line: 5, type: !12, isLocal: false, isDefinition: true, variable: i16* @g_118)
!14 = !{i32 2, !"Dwarf Version", i32 4}
!15 = !{i32 2, !"Debug Info Version", i32 3}
!16 = !{!"FlexASIC FlexC Compiler v6.38 for FADER (CoSy 6231.35) (LLVM)"}
!17 = distinct !DISubprogram(name: "S0", scope: !1, file: !1, line: 10, type: !18, isLocal: false, isDefinition: true, scopeLine: 10, isOptimized: false, unit: !0, variables: !2)
!18 = !DISubroutineType(types: !19)
!19 = !{!12}
!20 = !DILocation(line: 14, column: 3, scope: !21, inlinedAt: !27)
!21 = distinct !DISubprogram(name: "func_54", scope: !1, file: !1, line: 12, type: !22, isLocal: false, isDefinition: true, scopeLine: 12, isOptimized: false, unit: !0, variables: !2)
!22 = !DISubroutineType(types: !23)
!23 = !{!24, !12, !12}
!24 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint64_t", file: !6, line: 107, baseType: !25)
!25 = !DIDerivedType(tag: DW_TAG_typedef, name: "__u64_t", file: !1, baseType: !26)
!26 = !DIBasicType(name: "unsigned long long", size: 64, align: 16, encoding: DW_ATE_unsigned)
!27 = distinct !DILocation(line: 10, column: 8, scope: !17)
!28 = !DILocation(line: 8, column: 12, scope: !29, inlinedAt: !30)
!29 = distinct !DISubprogram(name: "func_9", scope: !1, file: !1, line: 8, type: !18, isLocal: false, isDefinition: true, scopeLine: 8, isOptimized: false, unit: !0, variables: !2)
!30 = distinct !DILocation(line: 14, column: 3, scope: !21, inlinedAt: !27)
!31 = !DILocation(line: 20, column: 8, scope: !32, inlinedAt: !39)
!32 = distinct !DISubprogram(name: "func_61", scope: !1, file: !1, line: 19, type: !33, isLocal: false, isDefinition: true, scopeLine: 19, isOptimized: false, unit: !0, variables: !2)
!33 = !DISubroutineType(types: !34)
!34 = !{!35, !36, !5, !35, !35}
!35 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !5, size: 16, align: 16)
!36 = !DIDerivedType(tag: DW_TAG_typedef, name: "uint32_t", file: !6, line: 105, baseType: !37)
!37 = !DIDerivedType(tag: DW_TAG_typedef, name: "__u32_t", file: !1, baseType: !38)
!38 = !DIBasicType(name: "unsigned long", size: 32, align: 16, encoding: DW_ATE_unsigned)
!39 = distinct !DILocation(line: 14, column: 3, scope: !21, inlinedAt: !27)
!40 = !DILocalVariable(name: "p_63", arg: 2, scope: !32, line: 19, type: !5)
!41 = !DIExpression()
!42 = !DILocation(line: 19, column: 41, scope: !32, inlinedAt: !39)
!43 = !DILocation(line: 15, column: 10, scope: !21, inlinedAt: !27)
!44 = !DILocation(line: 15, column: 20, scope: !21, inlinedAt: !27)
