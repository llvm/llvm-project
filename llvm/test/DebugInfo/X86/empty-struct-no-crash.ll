; RUN: llc -mtriple=x86_64-apple-darwin -o /dev/null %s
; Verify we don't crash on DISubprogram without a type.

target triple = "x86_64-apple-darwin"

define void @test() !dbg !4 {
  ret void
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 2, !"Dwarf Version", i32 4}
!4 = distinct !DISubprogram(name: "test", scope: !1, file: !1, line: 1, scopeLine: 1, spFlags: DISPFlagDefinition, unit: !0)