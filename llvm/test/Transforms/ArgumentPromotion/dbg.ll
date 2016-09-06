; RUN: opt < %s -argpromotion -S | FileCheck %s

declare void @sink(i32)

; CHECK: define internal void @test({{.*}} !dbg [[SP:![0-9]+]]
define internal void @test(i32** %X) !dbg !2 {
  %1 = load i32*, i32** %X, align 8
  %2 = load i32, i32* %1, align 8
  call void @sink(i32 %2)
  ret void
}

define void @caller(i32** %Y) {
; CHECK: call void @test(i32 %
  call void @test(i32** %Y)
  ret void
}

; CHECK: [[SP]] = distinct !DISubprogram(name: "test",

!llvm.module.flags = !{!0}
!llvm.dbg.cu = !{!3}

!0 = !{i32 2, !"Debug Info Version", i32 3}
!1 = !DILocation(line: 8, scope: !2)
!2 = distinct !DISubprogram(name: "test", line: 3, isLocal: true, isDefinition: true, virtualIndex: 6, flags: DIFlagPrototyped, isOptimized: false, unit: !3, scopeLine: 3, scope: null)
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, producer: "clang version 3.5.0 ", isOptimized: false, emissionKind: LineTablesOnly, file: !5)
!5 = !DIFile(filename: "test.c", directory: "")
