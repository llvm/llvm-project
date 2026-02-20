source_filename = "unit-3.cpp"
target triple = "x86_64-unknown-linux"

@Var_3 = dso_local global i32 1, align 4, !dbg !0

define dso_local noundef i32 @_Z5foo_3i(i32 noundef %P3) !dbg !14 {
entry:
  %P3.addr = alloca i32, align 4
  %V3 = alloca i32, align 4
  store i32 %P3, ptr %P3.addr, align 4
    #dbg_declare(ptr %P3.addr, !18, !DIExpression(), !19)
    #dbg_declare(ptr %V3, !20, !DIExpression(), !21)
  %0 = load i32, ptr %P3.addr, align 4, !dbg !22
  store i32 %0, ptr %V3, align 4, !dbg !21
  %1 = load i32, ptr %V3, align 4, !dbg !23
  %2 = load i32, ptr @Var_3, align 4, !dbg !24
  %add = add nsw i32 %1, %2, !dbg !25
  ret i32 %add, !dbg !26
}

define dso_local void @_Z2f3v() !dbg !27 {
entry:
  %call = call noundef i32 @_Z5foo_3i(i32 noundef 333), !dbg !30
  ret void, !dbg !31
}

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8}
!llvm.ident = !{!13}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "Var_3", scope: !2, file: !3, line: 1, type: !5, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !3, producer: "clang version 23.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, globals: !4, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "unit-3.cpp", directory: "", checksumkind: CSK_MD5, checksum: "fcd53200d3096d44f8a8308b6ed33c23")
!4 = !{!0}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 1, !"wchar_size", i32 4}
!13 = !{!"clang version 23.0.0"}
!14 = distinct !DISubprogram(name: "foo_3", linkageName: "_Z5foo_3i", scope: !3, file: !3, line: 2, type: !15, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !17)
!15 = !DISubroutineType(types: !16)
!16 = !{!5, !5}
!17 = !{}
!18 = !DILocalVariable(name: "P3", arg: 1, scope: !14, file: !3, line: 2, type: !5)
!19 = !DILocation(line: 2, column: 15, scope: !14)
!20 = !DILocalVariable(name: "V3", scope: !14, file: !3, line: 3, type: !5)
!21 = !DILocation(line: 3, column: 7, scope: !14)
!22 = !DILocation(line: 3, column: 12, scope: !14)
!23 = !DILocation(line: 4, column: 10, scope: !14)
!24 = !DILocation(line: 4, column: 15, scope: !14)
!25 = !DILocation(line: 4, column: 13, scope: !14)
!26 = !DILocation(line: 4, column: 3, scope: !14)
!27 = distinct !DISubprogram(name: "f3", linkageName: "_Z2f3v", scope: !3, file: !3, line: 7, type: !28, scopeLine: 7, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2)
!28 = !DISubroutineType(types: !29)
!29 = !{null}
!30 = !DILocation(line: 8, column: 3, scope: !27)
!31 = !DILocation(line: 9, column: 1, scope: !27)
