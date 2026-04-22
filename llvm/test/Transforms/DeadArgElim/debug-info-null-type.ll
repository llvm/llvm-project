; RUN: opt -S -passes=verify,deadargelim < %s | FileCheck %s

; Test that deadargelim does not crash when a DISubprogram has a null type.
; This is valid with line-table-only debug info, where no DISubroutineType may
; be available to mark with DW_CC_nocall.

define internal i32 @callee(i32 %used, i32 %dead) !dbg !4 {
; CHECK-LABEL: define internal i32 @callee(
; CHECK-SAME: i32 [[USED:%.*]]) !dbg [[SP:![0-9]+]]
entry:
  ret i32 %used
}

define i32 @caller(i32 %x) {
; CHECK-LABEL: define i32 @caller(
entry:
; CHECK: call i32 @callee(i32 %x)
  %r = call i32 @callee(i32 %x, i32 42), !dbg !3
  ret i32 %r
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!2 = !DIFile(filename: "test.c", directory: "")
!3 = !DILocation(line: 8, scope: !4)
; CHECK: [[SP]] = distinct !DISubprogram(name: "callee", scope: null, file: [[FILE:![0-9]+]], line: 3, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: [[CU:![0-9]+]])
!4 = distinct !DISubprogram(name: "callee", scope: null, file: !2, line: 3, type: null, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1)
