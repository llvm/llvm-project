source_filename = "unit.cpp"
target triple = "x86_64-unknown-linux"

define dso_local void @_Z3foov() !dbg !10 {
entry:
  call void @_Z2f1v(), !dbg !13
  call void @_Z2f2v(), !dbg !14
  call void @_Z2f3v(), !dbg !15
  ret void, !dbg !16
}

declare void @_Z2f1v()

declare void @_Z2f2v()

declare void @_Z2f3v()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 23.0.0", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "unit.cpp", directory: "", checksumkind: CSK_MD5, checksum: "d340f597b69d89b7c542f22ce3d36231")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!9 = !{!"clang version 23.0.0"}
!10 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !1, file: !1, line: 5, type: !11, scopeLine: 5, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 6, column: 3, scope: !10)
!14 = !DILocation(line: 7, column: 3, scope: !10)
!15 = !DILocation(line: 8, column: 3, scope: !10)
!16 = !DILocation(line: 9, column: 1, scope: !10)
