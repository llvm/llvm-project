target triple = "x86_64-unknown-linux-gnu"

define dso_local void @_Z1bi(i32 %i) local_unnamed_addr !dbg !7 {
entry:
  call void @llvm.dbg.value(metadata i32 %i, metadata !12, metadata !DIExpression()), !dbg !13
  tail call void asm sideeffect "", "~{rdi},~{dirflag},~{fpsr},~{flags}"() , !dbg !14, !srcloc !15
  ret void, !dbg !16
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3, !4, !5}
!llvm.ident = !{!6}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, producer: "clang version 8.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !2, nameTableKind: None)
!1 = !DIFile(filename: "b.cpp", directory: "/home/test/PRs/PR38990")
!2 = !{}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = !{i32 2, !"Debug Info Version", i32 3}
!5 = !{i32 1, !"wchar_size", i32 4}
!6 = !{!"clang version 8.0.0"}
!7 = distinct !DISubprogram(name: "b", linkageName: "_Z1bi", scope: !1, file: !1, line: 1, type: !8, isLocal: false, isDefinition: true, scopeLine: 1, flags: DIFlagPrototyped, isOptimized: true, unit: !0, retainedNodes: !11)
!8 = !DISubroutineType(types: !9)
!9 = !{null, !10}
!10 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!11 = !{!12}
!12 = !DILocalVariable(name: "i", arg: 1, scope: !7, file: !1, line: 1, type: !10)
!13 = !DILocation(line: 1, column: 12, scope: !7)
!14 = !DILocation(line: 1, column: 17, scope: !7)
!15 = !{i32 22}
!16 = !DILocation(line: 1, column: 38, scope: !7)
