; REQUIRES: object-emission

; RUN: %llc_dwarf -O0 -filetype=obj %s -o - | llvm-dwarfdump -v -debug-info - | FileCheck %s
; Use correct signedness when emitting constants of derived (sugared) types.

; CHECK: DW_AT_const_value [DW_FORM_sdata] (42)
; CHECK: DW_AT_const_value [DW_FORM_udata] (117)
; CHECK: DW_AT_const_value [DW_FORM_udata] (7)

; Function Attrs: uwtable
define void @main() #0 !dbg !4 {
entry:
  tail call void @llvm.dbg.value(metadata i32 42, metadata !10, metadata !DIExpression()), !dbg !21
  tail call void @llvm.dbg.value(metadata i32 117, metadata !12, metadata !DIExpression()), !dbg !24
  tail call void @llvm.dbg.value(metadata i16 7, metadata !15, metadata !DIExpression()), !dbg !27
  ret void, !dbg !29
}

; Function Attrs: nounwind readnone
declare void @llvm.dbg.value(metadata, metadata, metadata) #2

attributes #0 = { uwtable }
attributes #2 = { nounwind readnone }

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!17, !18}
!llvm.ident = !{!19}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: true, emissionKind: FullDebug, file: !1, enums: !2, retainedTypes: !2, globals: !2, imports: !2)
!1 = !DIFile(filename: "const.cpp", directory: "/tmp/dbginfo")
!2 = !{}
!4 = distinct !DISubprogram(name: "main", line: 4, isLocal: false, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: true, unit: !0, scopeLine: 4, file: !1, scope: !5, type: !6, retainedNodes: !9)
!5 = !DIFile(filename: "const.cpp", directory: "/tmp/dbginfo")
!6 = !DISubroutineType(types: !7)
!7 = !{!8}
!8 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!9 = !{!10, !12, !15}
!10 = !DILocalVariable(name: "i", line: 5, scope: !4, file: !5, type: !11)
!11 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !8)
!12 = !DILocalVariable(name: "j", line: 7, scope: !4, file: !5, type: !13)
!13 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !14)
!14 = !DIBasicType(tag: DW_TAG_base_type, name: "unsigned int", size: 32, align: 32, encoding: DW_ATE_unsigned)
!15 = !DILocalVariable(name: "c", line: 9, scope: !4, file: !5, type: !16)
!16 = !DIBasicType(tag: DW_TAG_base_type, name: "char16_t", size: 16, align: 16, encoding: 16)
!17 = !{i32 2, !"Dwarf Version", i32 4}
!18 = !{i32 1, !"Debug Info Version", i32 3}
!19 = !{!"clang version 3.5.0 "}
!20 = !{i32 42}
!21 = !DILocation(line: 5, scope: !4)
!22 = !DILocation(line: 6, scope: !4)
!23 = !{i32 117}
!24 = !DILocation(line: 7, scope: !4)
!25 = !DILocation(line: 8, scope: !4)
!26 = !{i16 7}
!27 = !DILocation(line: 9, scope: !4)
!28 = !DILocation(line: 10, scope: !4)
!29 = !DILocation(line: 11, scope: !4)
