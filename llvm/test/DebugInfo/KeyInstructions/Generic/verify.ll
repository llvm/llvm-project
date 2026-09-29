; RUN: opt %s -o - -S --passes=verify 2>&1 | FileCheck %s

; CHECK: DbgLoc uses atomGroup but DISubprogram doesn't have Key Instructions enabled
; CHECK-NEXT: ![[#]] = !DILocation(line: 1, column: 11, scope: ![[f:.*]], atomGroup: 1, atomRank: 1)
; CHECK-NEXT: ![[f]] = distinct !DISubprogram(name: "f"
; CHECK-NEXT: warning: ignoring invalid debug info

define dso_local void @f() !dbg !10 {
entry:
; Include non-key location to check verifier is checking the whole function.
  %0 = add i32 0, 0, !dbg !14
  ret void, !dbg !13
}

!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1, producer: "clang version 21.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.cpp", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!9 = !{!"clang version 21.0.0git"}
!10 = distinct !DISubprogram(name: "f", scope: !1, file: !1, line: 1, type: !11, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!11 = !DISubroutineType(types: !12)
!12 = !{null}
!13 = !DILocation(line: 1, column: 11, scope: !10, atomGroup: 1, atomRank: 1)
!14 = !DILocation(line: 1, column: 11, scope: !10)
