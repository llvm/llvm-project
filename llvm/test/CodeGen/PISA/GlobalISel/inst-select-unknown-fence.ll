; RUN: not llc -mtriple=pisa -stop-after=instruction-select -filetype=null %s -o /dev/null 2>&1 | FileCheck %s

source_filename = "unknown-fence.c"
target triple = "pisa"

define void @test_unknown_fence() !dbg !4 {
  fence syncscope("foo") seq_cst, !dbg !8
  ret void
}

; CHECK: error: unknown-fence.c:7:3: G_FENCE syncscope is not supported on PISA

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!1}
!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !2, producer: "test", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug)
!1 = !{i32 2, !"Debug Info Version", i32 3}
!2 = !DIFile(filename: "unknown-fence.c", directory: "/tmp")
!4 = distinct !DISubprogram(name: "test_unknown_fence", scope: !2, file: !2, line: 1, type: !5, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{}
!8 = !DILocation(line: 7, column: 3, scope: !4)
