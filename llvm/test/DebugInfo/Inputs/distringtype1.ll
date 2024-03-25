define void @sub2_(ptr %string, i64 %.U0001.arg) !dbg !5 {
L.entry:
  call void @llvm.dbg.declare(metadata ptr %string, metadata !12, metadata !DIExpression()), !dbg !14
  call void @llvm.dbg.value(metadata i64 %.U0001.arg, metadata !10, metadata !DIExpression()), !dbg !14
  ret void, !dbg !15
}

declare void @llvm.dbg.declare(metadata, metadata, metadata)
declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.module.flags = !{!0, !1}
!llvm.dbg.cu = !{!2}

!0 = !{i32 2, !"Dwarf Version", i32 4}
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = distinct !DICompileUnit(language: DW_LANG_Fortran90, file: !3, producer: " F90 Flang - 1.5 2017-05-01", isOptimized: false, flags: "'+flang -g distringtype1.f90 -S -emit-llvm'", runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !4, globals: !4, imports: !4, nameTableKind: None)
!3 = !DIFile(filename: "distringtype1.f90", directory: "/tmp/")
!4 = !{}
!5 = distinct !DISubprogram(name: "sub2", scope: !2, file: !3, line: 1, type: !6, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !9)
!6 = !DISubroutineType(types: !7)
!7 = !{null, !8}
!8 = !DIStringType(name: "character(*)!1", size: 32)
!9 = !{!10, !12}
!10 = !DILocalVariable(arg: 2, scope: !5, file: !3, line: 1, type: !11, flags: DIFlagArtificial)
!11 = !DIBasicType(name: "integer*8", size: 64, align: 64, encoding: DW_ATE_signed)
!12 = !DILocalVariable(name: "string", arg: 1, scope: !5, file: !3, line: 1, type: !13)
!13 = !DIStringType(name: "character(*)!2", stringLength: !10, stringLengthExpression: !DIExpression(), size: 32)
!14 = !DILocation(line: 0, scope: !5)
!15 = !DILocation(line: 3, column: 1, scope: !5)
