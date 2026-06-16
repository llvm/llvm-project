; RUN: opt %s -o - -S| FileCheck %s

; CHECK: distinct !DISubprogram(name: "f", {{.*}}keyInstructions: true)
; CHECK: !DILocation(line: 1, column: 11, scope: ![[#]], atomGroup: 1, atomRank: 1)

define dso_local void @f() !dbg !10 {
entry:
  ret void, !dbg !13
}

!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 21.0.0git"}
!10 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, keyInstructions: true)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 1, column: 11, scope: !10, atomGroup: 1, atomRank: 1)
