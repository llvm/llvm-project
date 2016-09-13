; RUN: llc -mtriple mips-linux-gnu -O2 %s -o - | FileCheck %s
; Check struct X for dead variable xyz from inlined function foo.

; CHECK: section_info
; CHECK:	DW_TAG_structure_type
; CHECK-NEXT:	info_string


@i = common global i32 0, !dbg !16                          ; <i32*> [#uses=2]

declare void @llvm.dbg.declare(metadata, metadata, metadata) nounwind readnone

declare void @llvm.dbg.value(metadata, i64, metadata, metadata) nounwind readnone

define i32 @bar() nounwind ssp !dbg !6 {
entry:
  %0 = load i32, i32* @i, align 4, !dbg !17            ; <i32> [#uses=2]
  tail call void @llvm.dbg.value(metadata i32 %0, i64 0, metadata !59, metadata !DIExpression()), !dbg !19
  tail call void @llvm.dbg.declare(metadata !29, metadata !60, metadata !DIExpression()), !dbg !21
  %1 = mul nsw i32 %0, %0, !dbg !22               ; <i32> [#uses=2]
  store i32 %1, i32* @i, align 4, !dbg !17
  ret i32 %1, !dbg !23
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!28}

!0 = distinct !DISubprogram(name: "foo", line: 9, isLocal: true, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !2, scopeLine: 9, file: !27, scope: !1, type: !3, variables: !24)
!1 = !DIFile(filename: "bar.c", directory: "/tmp/")
!2 = distinct !DICompileUnit(language: DW_LANG_C89, producer: "4.2.1 (Based on Apple Inc. build 5658) (LLVM build)", isOptimized: true, emissionKind: FullDebug, file: !27, enums: !20, retainedTypes: !20, globals: !26, imports:  !20)
!3 = !DISubroutineType(types: !4)
!4 = !{!5, !5}
!5 = !DIBasicType(tag: DW_TAG_base_type, name: "int", size: 32, align: 32, encoding: DW_ATE_signed)
!6 = distinct !DISubprogram(name: "bar", linkageName: "bar", line: 14, isLocal: false, isDefinition: true, virtualIndex: 6, isOptimized: true, unit: !2, file: !27, scope: !1, type: !7)
!7 = !DISubroutineType(types: !8)
!8 = !{!5}
!9 = !DILocalVariable(name: "j", line: 9, arg: 1, scope: !0, file: !1, type: !5)
!10 = !DILocalVariable(name: "xyz", line: 10, scope: !11, file: !1, type: !12)

!59 = !DILocalVariable(name: "j", line: 9, arg: 1, scope: !0, file: !1, type: !5)
!60 = !DILocalVariable(name: "xyz", line: 10, scope: !11, file: !1, type: !12)

!11 = distinct !DILexicalBlock(line: 9, column: 0, file: !1, scope: !0)
!12 = !DICompositeType(tag: DW_TAG_structure_type, name: "X", line: 10, size: 64, align: 32, file: !27, scope: !0, elements: !13)
!13 = !{!14, !15}
!14 = !DIDerivedType(tag: DW_TAG_member, name: "a", line: 10, size: 32, align: 32, file: !27, scope: !12, baseType: !5)
!15 = !DIDerivedType(tag: DW_TAG_member, name: "b", line: 10, size: 32, align: 32, offset: 32, file: !27, scope: !12, baseType: !5)
!16 = !DIGlobalVariable(name: "i", line: 5, isLocal: false, isDefinition: true, scope: !1, file: !1, type: !5)
!17 = !DILocation(line: 15, scope: !18)
!18 = distinct !DILexicalBlock(line: 14, column: 0, file: !1, scope: !6)
!19 = !DILocation(line: 9, scope: !0, inlinedAt: !17)
!20 = !{}
!21 = !DILocation(line: 9, scope: !11, inlinedAt: !17)
!22 = !DILocation(line: 11, scope: !11, inlinedAt: !17)
!23 = !DILocation(line: 16, scope: !18)
!24 = !{!9, !10}
!26 = !{!16}
!27 = !DIFile(filename: "bar.c", directory: "/tmp/")
!28 = !{i32 1, !"Debug Info Version", i32 3}
!29 = !{null}
