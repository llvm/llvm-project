; RUN: opt < %s -S | FileCheck %s

source_filename = "/llvm-project/foo.c"

define dso_local void @f() !dbg !10 {
entry:
  ret void, !dbg !14
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3, !4, !5, !6, !7, !8}
!llvm.ident = !{!9}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "/llvm-project/foo.c", directory: "/llvm-project")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !{i32 1, !"wchar_size", i32 4}
!5 = !{i32 8, !"PIC Level", i32 2}
!6 = !{i32 7, !"PIE Level", i32 2}
!7 = !{i32 7, !"uwtable", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{!"clang"}
!10 = distinct !DISubprogram(name: "f", scope: !11, file: !11, line: 1, type: !12, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DIFile(filename: "foo.c", directory: "/llvm-project")
!12 = !DISubroutineType(types: !13)
!13 = !{null}
!14 = !DILocation(line: 1, column: 11, scope: !10)
