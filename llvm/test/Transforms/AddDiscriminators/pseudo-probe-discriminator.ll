; RUN: opt < %s -passes=add-discriminators -S | FileCheck %s

; Test that AddDiscriminators preserves pseudo-probe discriminators.
; Pseudo-probe discriminators use bits [2:0] = 0x7 as a marker.

define void @foo() !dbg !4 {
entry:
  call void @bar(), !dbg !10
  call void @bar(), !dbg !11
  ret void
}

declare void @bar()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!7, !8}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus, file: !1, isOptimized: false, runtimeVersion: 0, emissionKind: NoDebug)
!1 = !DIFile(filename: "test.cpp", directory: "/tmp")
!4 = distinct !DISubprogram(name: "foo", scope: !1, file: !1, line: 1, type: !5, scopeLine: 1, unit: !0)
!5 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !{i32 2, !"Dwarf Version", i32 4}
!8 = !{i32 2, !"Debug Info Version", i32 3}

; Two calls on the same line with pseudo-probe discriminators (bits [2:0] = 0x7)
!9 = !DILexicalBlockFile(scope: !4, file: !1, discriminator: 455081999)
!10 = !DILocation(line: 2, column: 3, scope: !9)
!12 = !DILexicalBlockFile(scope: !4, file: !1, discriminator: 455082007)
!11 = !DILocation(line: 2, column: 3, scope: !12)

; CHECK-DAG: !DILexicalBlockFile({{.*}}discriminator: 455081999)
; CHECK-DAG: !DILexicalBlockFile({{.*}}discriminator: 455082007)
