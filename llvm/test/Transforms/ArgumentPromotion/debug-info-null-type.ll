; RUN: opt -S -passes=verify,argpromotion < %s | FileCheck %s

; Test that argpromotion does not crash when a DISubprogram has a null type.
; This is valid with line-table-only debug info, where no DISubroutineType may
; be available to mark with DW_CC_nocall.
; https://github.com/llvm/llvm-project/issues/186557

%struct.pair = type { i32, i32 }

define internal void @test(ptr %X) !dbg !4 {
; CHECK-LABEL: define internal void @test(
; CHECK-SAME: i32 %{{.*}}) !dbg [[SP:![0-9]+]]
  %1 = load ptr, ptr %X, align 8
  %2 = load i32, ptr %1, align 8
  call void @sink(i32 %2)
  ret void
}

declare void @sink(i32)

define void @caller(ptr %Y, ptr %P) {
; CHECK-LABEL: define void @caller(
  call void @test(ptr %Y), !dbg !3
  ret void
}

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!1}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !2, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!2 = !DIFile(filename: "test.c", directory: "")
!3 = !DILocation(line: 8, scope: !4)
; CHECK: [[SP]] = distinct !DISubprogram(name: "test", scope: null, file: [[FILE:![0-9]+]], line: 3, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: [[CU:![0-9]+]])
!4 = distinct !DISubprogram(name: "test", scope: null, file: !2, line: 3, type: null, scopeLine: 3, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !1)
