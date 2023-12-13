; ModuleID = 'argc_sub1.c'

define i32 @sub1(i32) !dbg !8 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !13, metadata !DIExpression()), !dbg !14
  %2 = add nsw i32 %0, -1, !dbg !15
  ret i32 %2, !dbg !16
}

define i32 @main(i32, i8** nocapture readnone) !dbg !17 {
  call void @llvm.dbg.value(metadata i32 %0, metadata !24, metadata !DIExpression()), !dbg !26
  call void @llvm.dbg.value(metadata i8** %1, metadata !25, metadata !DIExpression()), !dbg !27
  %3 = tail call i32 @sub1(i32 %0), !dbg !28
  ret i32 %3, !dbg !29
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5, !6}
!llvm.ident = !{!7}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang version 7.0.1-8+deb10u2 (tags/RELEASE_701/final)", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2)
!1 = !DIFile(filename: "argc_sub1.c", directory: "Inputs/")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{i32 7, !"PIC Level", i32 2}
!7 = !{!"clang version 7.0.1-8+deb10u2 (tags/RELEASE_701/final)"}
!8 = distinct !DISubprogram(name: "sub1", scope: !1, file: !1, line: 1, type: !9, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !12)
!9 = !DISubroutineType(types: !10)
!10 = !{!11, !11}
!11 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!12 = !{!13}
!13 = !DILocalVariable(name: "x", arg: 1, scope: !8, file: !1, line: 1, type: !11)
!14 = !DILocation(line: 1, column: 14, scope: !8)
!15 = !DILocation(line: 1, column: 28, scope: !8)
!16 = !DILocation(line: 1, column: 19, scope: !8)
!17 = distinct !DISubprogram(name: "main", scope: !1, file: !1, line: 2, type: !18, isLocal: false, isDefinition: true, scopeLine: 2, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !23)
!18 = !DISubroutineType(types: !19)
!19 = !{!11, !11, !20}
!20 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !21, size: 64)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !22, size: 64)
!22 = !DIBasicType(name: "char", size: 8, encoding: DW_ATE_signed_char)
!23 = !{!24, !25}
!24 = !DILocalVariable(name: "argc", arg: 1, scope: !17, file: !1, line: 2, type: !11)
!25 = !DILocalVariable(name: "argv", arg: 2, scope: !17, file: !1, line: 2, type: !20)
!26 = !DILocation(line: 2, column: 14, scope: !17)
!27 = !DILocation(line: 2, column: 27, scope: !17)
!28 = !DILocation(line: 2, column: 42, scope: !17)
!29 = !DILocation(line: 2, column: 35, scope: !17)
