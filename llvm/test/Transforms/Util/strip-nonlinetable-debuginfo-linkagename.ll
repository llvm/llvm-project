; RUN: opt -S -passes=strip-nonlinetable-debuginfo %s -o - | FileCheck %s

define void @_Z3foov() !dbg !4 {
  ret void, !dbg !6
}

define void @_Z3barv() !dbg !7 {
  ret void, !dbg !8
}

define void @_Z3bazv() !dbg !9 {
  ret void, !dbg !10
}

; CHECK: define void @_Z3foov() !dbg ![[FOO:[0-9]+]]
; CHECK: define void @_Z3barv() !dbg ![[BAR:[0-9]+]]
; CHECK: define void @_Z3bazv() !dbg ![[BAZ:[0-9]+]]
; CHECK: ![[PROFILING_CU:[0-9]+]] = distinct !DICompileUnit({{.*}}emissionKind: LineTablesOnly, debugInfoForProfiling: true)
; CHECK: ![[REGULAR_CU:[0-9]+]] = distinct !DICompileUnit({{.*}}emissionKind: LineTablesOnly)
; CHECK: ![[FOO]] = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", {{.*}}unit: ![[PROFILING_CU]])
; CHECK: ![[BAR]] = distinct !DISubprogram(name: "bar", scope:
; CHECK-NOT: linkageName:
; CHECK-SAME: unit: ![[REGULAR_CU]])
; CHECK: ![[BAZ]] = distinct !DISubprogram(linkageName: "_Z3bazv", {{.*}}unit: ![[REGULAR_CU]])

!llvm.dbg.cu = !{!0, !1}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, emissionKind: FullDebug, debugInfoForProfiling: true)
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, emissionKind: FullDebug)
!2 = !DIFile(filename: "foo.cpp", directory: "/")
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = distinct !DISubprogram(name: "foo", linkageName: "_Z3foov", scope: !2, file: !2, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !{})
!6 = !DILocation(line: 1, column: 1, scope: !4)
!7 = distinct !DISubprogram(name: "bar", linkageName: "_Z3barv", scope: !2, file: !2, line: 2, type: !5, scopeLine: 2, spFlags: DISPFlagDefinition, unit: !1)
!8 = !DILocation(line: 2, column: 1, scope: !7)
!9 = distinct !DISubprogram(linkageName: "_Z3bazv", scope: !2, file: !2, line: 3, type: !5, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !1)
!10 = !DILocation(line: 3, column: 1, scope: !9)
