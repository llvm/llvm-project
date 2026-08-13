; REQUIRES: x86-registered-target
; RUN: llc -mtriple=x86_64-unknown-linux-gnu -filetype=asm %s -o - | FileCheck %s

; An inline-tree edge is valid only when its inlined-at location carries a
; packed pseudo-probe discriminator. A zero discriminator must not manufacture
; an inline site at probe zero. If a nested chain has a valid inner edge and an
; invalid outer edge, retain the inner edge and truncate the chain there.

; CHECK:      .pseudoprobe 101 1 0 0 invalid
; CHECK-NEXT: .pseudoprobe 102 1 0 0 invalid
; CHECK:      .pseudoprobe 103 1 0 0 @ {{[0-9]+}}:2 valid

@value = external global i32

define void @invalid() !dbg !5 {
entry:
  call void @llvm.pseudoprobe(i64 101, i64 1, i32 0, i64 -1), !dbg !16
  %v = load volatile i32, ptr @value, align 4, !dbg !17
  call void @llvm.pseudoprobe(i64 102, i64 1, i32 0, i64 -1), !dbg !18
  store volatile i32 %v, ptr @value, align 4, !dbg !19
  ret void, !dbg !20
}

define void @valid() !dbg !6 {
entry:
  call void @llvm.pseudoprobe(i64 103, i64 1, i32 0, i64 -1), !dbg !21
  %v = load volatile i32, ptr @value, align 4, !dbg !22
  ret void, !dbg !23
}

declare void @llvm.pseudoprobe(i64, i64, i32, i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!llvm.pseudo_probe_desc = !{!12, !13, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_14, file: !1,
                             producer: "pseudo-probe-invalid-inline-site",
                             isOptimized: true, runtimeVersion: 0,
                             emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "inline.cpp", directory: "/tmp")
!2 = !{i32 7, !"Dwarf Version", i32 4}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !15)
!5 = distinct !DISubprogram(name: "invalid", linkageName: "invalid", scope: !1,
                            file: !1, line: 1, type: !4, scopeLine: 1,
                            spFlags: DISPFlagDefinition | DISPFlagOptimized,
                            unit: !0)
!6 = distinct !DISubprogram(name: "valid", linkageName: "valid", scope: !1,
                            file: !1, line: 5, type: !4, scopeLine: 5,
                            spFlags: DISPFlagDefinition | DISPFlagOptimized,
                            unit: !0)
!7 = distinct !DISubprogram(name: "leaf_one", linkageName: "leaf_one",
                            scope: !1, file: !1, line: 10, type: !4,
                            scopeLine: 10,
                            spFlags: DISPFlagDefinition | DISPFlagOptimized,
                            unit: !0)
!8 = distinct !DISubprogram(name: "leaf_two", linkageName: "leaf_two",
                            scope: !1, file: !1, line: 20, type: !4,
                            scopeLine: 20,
                            spFlags: DISPFlagDefinition | DISPFlagOptimized,
                            unit: !0)
!9 = distinct !DISubprogram(name: "leaf_three", linkageName: "leaf_three",
                            scope: !1, file: !1, line: 30, type: !4,
                            scopeLine: 30,
                            spFlags: DISPFlagDefinition | DISPFlagOptimized,
                            unit: !0)
!10 = distinct !DILocation(line: 2, column: 3, scope: !5)
!11 = !DILexicalBlockFile(scope: !25, file: !1, discriminator: 455082007)
!12 = !{i64 101, i64 1, !"leaf_one"}
!13 = !{i64 102, i64 1, !"leaf_two"}
!14 = !{i64 103, i64 1, !"leaf_three"}
!15 = !{}
!16 = !DILocation(line: 10, column: 3, scope: !7, inlinedAt: !10)
!17 = !DILocation(line: 11, column: 3, scope: !7, inlinedAt: !10)
!18 = !DILocation(line: 20, column: 3, scope: !8, inlinedAt: !10)
!19 = !DILocation(line: 21, column: 3, scope: !8, inlinedAt: !10)
!20 = !DILocation(line: 3, column: 1, scope: !5)
!21 = !DILocation(line: 30, column: 3, scope: !9, inlinedAt: !24)
!22 = !DILocation(line: 31, column: 3, scope: !9, inlinedAt: !24)
!23 = !DILocation(line: 7, column: 1, scope: !6)
!24 = distinct !DILocation(line: 41, column: 3, scope: !11, inlinedAt: !26)
!25 = distinct !DISubprogram(name: "middle", linkageName: "middle", scope: !1,
                             file: !1, line: 40, type: !4, scopeLine: 40,
                             spFlags: DISPFlagDefinition | DISPFlagOptimized,
                             unit: !0)
!26 = distinct !DILocation(line: 6, column: 3, scope: !6)
